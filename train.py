from config import Config
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from datasets.dataset import UrbanSoundDataset
from networks.cnn import CNNNetwork

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs,targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagation and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss : {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("--------------------------------------------------")
    print("Training is done")


if __name__ == "__main__":

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # instantiate dataset object and create data loader
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(Config.ANNOTATIONS_FILE,
                            Config.AUDIO_DIRECTORY,
                            mel_spectogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    # create data loader
    train_data_loader = create_data_loader(usd, batch_size=BATCH_SIZE)

    # Build Model

    print(f"Using Device {device}")
    cnn_net = CNNNetwork().to(device=device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn_net.parameters(), lr=LEARNING_RATE)

    # Train model
    train(cnn_net, train_data_loader, loss_fn=loss_fn, optimizer=optimiser, device=device, epochs=EPOCHS)

    torch.save(cnn_net.state_dict(), "feedforwardnet.pth")

    print("Model trained and saved at feedforwardnet.pth")
