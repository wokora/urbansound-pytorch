from config import Config
import torch.cuda
import torchaudio.transforms

from dataset import UrbanSoundDataset

if __name__ == "__main__":

    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'

    print(f"Using Device {device}")

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

    print(f'There are {len(usd)} samples')

    signal, label = usd[1]

    a = 1
