import torch
from torch import nn
import numpy as np
import soundfile as sf
import librosa

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

def preprocess_audio(file_path, n_fft=1024, hop_length=512):
    audio, sr = sf.read(file_path)
    audio_mono = librosa.to_mono(np.transpose(audio)) if audio.ndim > 1 else audio
    stft = librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    magnitude_db = librosa.amplitude_to_db(magnitude)
    # Reshape magnitude to match model input, handling the reshape more carefully
    magnitude_db_slices = [magnitude_db[:, i:i+1024] for i in range(0, magnitude_db.shape[1], 1024)]
    return magnitude_db_slices, phase, sr, stft.shape

def predict(model, magnitude_db_slices, device='cpu'):
    pred_magnitude_db_slices = []
    for slice in magnitude_db_slices:
        if slice.shape[1] < 1024:
            padding = np.zeros((slice.shape[0], 1024 - slice.shape[1]))
            slice = np.hstack((slice, padding))
        with torch.no_grad():
            slice_tensor = torch.tensor(slice, dtype=torch.float32).to(device)
            pred_slice = model(slice_tensor.unsqueeze(0)).squeeze(0)
        pred_magnitude_db_slices.append(pred_slice.cpu().numpy()[:, :slice.shape[1]])
    return pred_magnitude_db_slices

def postprocess_audio(pred_magnitude_db_slices, phase, sr, file_path_out, stft_shape, n_fft=1024, hop_length=512):
    # Concatenate predicted slices and trim to original STFT shape
    pred_magnitude_db = np.hstack(pred_magnitude_db_slices)[:stft_shape[0], :stft_shape[1]]
    pred_magnitude = librosa.db_to_amplitude(pred_magnitude_db)
    stft_matrix = pred_magnitude * phase
    audio_reconstructed = librosa.istft(stft_matrix, hop_length=hop_length)
    sf.write(file_path_out, audio_reconstructed, sr)

def enhance_speech(model_path, input_file_path, output_file_path, device='cpu'):
    model = load_model(model_path, device=device)
    magnitude_db_slices, phase, sr, stft_shape = preprocess_audio(input_file_path)
    pred_magnitude_db_slices = predict(model, magnitude_db_slices, device=device)
    postprocess_audio(pred_magnitude_db_slices, phase, sr, output_file_path, stft_shape)

# Example usage
if __name__ == "__main__":
    device = "cpu"
    audioPath = 'Test/test.flac'
    modelPath = 'model.pth'
    outputPath = 'separated_speech.wav'
    enhance_speech(modelPath, audioPath, outputPath)