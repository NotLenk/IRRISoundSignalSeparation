import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import dataprocessing as SL
import numpy as np
import sys

class MyModel(nn.Module):
    def __init__(self,input_channels):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_channels, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, input_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

def load_model(model_path, device):
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def stft(audio_data, n_fft=1024):
    # Compute the Short-Time Fourier Transform (STFT)
    stft_result = np.abs(np.fft.rfft(audio_data, n=n_fft))
    return stft_result

class CustomSoundDataset(Dataset):
    def __init__(self):
        self.speechData = pd.read_csv('Speech_Train.csv')
        self.noiseData = pd.read_csv('Noise_Train.csv')

    def __len__(self):
        return len(self.speechData) + len(self.noiseData)

    def __getitem__(self, _):
        speech_row = self.speechData.sample(n=1)
        noise_row = self.noiseData.sample(n=1)

        speech_path = speech_row['filepath'].values[0]
        noise_path = noise_row['filepath'].values[0]

        speech, _ = sf.read(speech_path)
        mixed = SL.snr_mixer(speech_path, noise_path)

        speechMagnitude = stft(speech)
        mixedMagnitude = stft(mixed)

        return speechMagnitude, mixedMagnitude


def magnitude_to_tensor(sound_wave, n_fft=1024):
    return sound_wave

def log(message):
    print(message)
    original_output = sys.stdout
    with open("Latest Log.txt", 'w') as file:
        sys.stdout = file
        print(message)
    sys.stdout = original_output

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for speechMagnitude, mixedMagnitude in dataloader:
        mixedMagnitude = mixedMagnitude.to(device).float()
        speechMagnitude = speechMagnitude.to(device).float()

        mixedTensor = magnitude_to_tensor(mixedMagnitude)
        speechTensor = magnitude_to_tensor(speechMagnitude)

        pred_magnitude = model(mixedTensor)
        loss = loss_fn(pred_magnitude, speechTensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss += loss.item()

    message = (f'Average Loss: {total_loss / len(dataloader)}')
    log(message)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csd = CustomSoundDataset()
    train_dataloader = DataLoader(csd, batch_size=1, shuffle=True)
    model = MyModel(input_channels=513).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.000001)
    epochs = 1

    doTraining = True
    if doTraining:
        for t in range(epochs):
            message = f"Epoch {t + 1}\n-------------------------------"
            log(message)
            train(model, train_dataloader, loss_fn, optimizer)
            torch.save(model.state_dict(), "model6.pth")
            # Model: 0      1      2       3        4       5 (acesta este un model ciudat)
            # Input: 1024   1024   4096    2048     1024    513
            # Batch: 10     5      5       5        10      1
            # Epoch: 10     10     30      30       250     20
            # Loss : ~42    ~40    ~74     ~126     ~29     ~1.2e-05
        print("Train done!")
