import torch
from torch import nn
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cpu"


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        x = None

        return x

class CustomSoundDataset(Dataset):
    def __init__(
        self, voice_annotations_file, noise_annotation_file, voice_dir, noise_dir
    ):
        pass

    # lungimea datasetului = nr. de samples = 1000
    def __len__(self):
        number_of_example = os.listdir()
        return number_of_example

    def __getitem__(self, index):
        x, y = None, None


# stft transform - use it if needed
def stft_transform(signal):
    return torch.stft(signal)
    pass


# istft transform - use it if needed
def istft_transform(spectrogram):
    return torch.istft(spectrogram)
    pass


# use if needed
def compute_magnitude(complex_signal):
    magnitudine = torch.sqrt(parte_reala ** 2 + parte_imaginara ** 2)
    pass


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # data preprocessing and model forward
        pred, targets = None, None
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
    # to be completed


if __name__ == "__main__":
    csd = CustomSoundDataset()
    train_dataloader = DataLoader(csd, batch_size=10, shuffle=False)
    model = MyModel.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    epochs = 20

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(model, train_dataloader, loss_fn, optimizer)
        torch.save(model.state_dict(), "model.pth")
    print("Train done!")
