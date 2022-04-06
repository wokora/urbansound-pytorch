import torch
from cnn import CNNNetwork
from torchsummary import summary

if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'

    cnn = CNNNetwork()
    summary( cnn.to(device), (1, 64, 44) )