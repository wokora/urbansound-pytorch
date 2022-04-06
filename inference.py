from config import Config
from train import SAMPLE_RATE, NUM_SAMPLES
import torch
import torchaudio
from networks.cnn import CNNNetwork
from datasets.dataset import UrbanSoundDataset

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model

    cnn = CNNNetwork().to(device)
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict=state_dict)

    # load UrbanSoundDataset validation dataset

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


    # get a sample from urban sound dataset for inference
    input, target = usd[0][0], usd[0][1]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted {predicted}, Expected {expected}")
