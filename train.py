from config import Config
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from datasets.dataset import UrbanSoundDataset
from networks.cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Using Device {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(Config.ANNOTATIONS_FILE,
                            Config.AUDIO_DIRECTORY,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    # create data loader
    train_data_loader = create_data_loader(usd, batch_size=BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
