import os


class Config:
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    USER_DIRECTORY = os.path.expanduser('~')
    DATA_DIRECTORY = os.path.join(USER_DIRECTORY, 'Datasets', 'UrbanSound8K')
    ANNOTATIONS_FILE = os.path.join(DATA_DIRECTORY, 'metadata', 'UrbanSound8K.csv')
    AUDIO_DIRECTORY = os.path.join(DATA_DIRECTORY, 'audio')
    MODEL_NAME = 'newtwork_name.pth'
