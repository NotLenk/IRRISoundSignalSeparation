import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import dataprocessing as SL

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        s1 = 1024
        s2 = 512
        self.layer1 = nn.Linear(in_features=s1, out_features=s2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=s2, out_features=s1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def load_model(model_path, device='cpu'):
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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

        mixedaudio = SL.snr_mixer(speech_path,noise_path)

        speech = torch.tensor(speech, dtype=torch.float32)
        mixedaudio = torch.tensor(mixedaudio, dtype=torch.float32)

        return speech, mixedaudio


def compute_magnitude(sound_wave, n_fft=1024):
    stft_result = torch.stft(sound_wave, n_fft, return_complex=True)
    magnitude = torch.abs(stft_result)
    # reshape tensor and keep 1024 features
    magnitude = magnitude.reshape(magnitude.size(0), -1)
    magnitude = magnitude[:, :1024] # features to 1024
    return magnitude



def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for mixed_audio, speech in dataloader:
        mixed_audio = mixed_audio.to(device)
        speech = speech.to(device)

        mixed_audio_magnitude = compute_magnitude(mixed_audio)
        speech_magnitude = compute_magnitude(speech)

        pred_magnitude = model(mixed_audio_magnitude)
        loss = loss_fn(pred_magnitude, speech_magnitude)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Average Loss: {total_loss / len(dataloader)}')

if __name__ == "__main__":
    device = "cpu"
    csd = CustomSoundDataset()
    train_dataloader = DataLoader(csd, batch_size=5, shuffle=True)
    model = MyModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    epochs = 10

    doTraining = True
    if doTraining:
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(model, train_dataloader, loss_fn, optimizer)
            torch.save(model.state_dict(), "model1.pth")
            # Model: 0      1      2       3
            # Input: 1024   1024   4096    2048
            # Batch: 10     5      5       5
            # Epoch: 10     10     30      30
            # Loss : ~42    ~40    ~74     ~126
        print("Train done!")