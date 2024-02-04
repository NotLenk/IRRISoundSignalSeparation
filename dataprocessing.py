import os
import librosa
import numpy as np
import glob
import soundfile as sf

# RMS = Metoda de masurare a volumului sunetului in medie
# SNR = Sound to Noise Ratio (Ratio intre speech si zgomot) (Cresti sau reduci cu cresterea volumului de zgomot)

def normalize(sound, target_level):
    rms = np.sqrt(np.mean(sound**2))
    desired_rms = 10**(target_level / 20)
    gain = desired_rms / rms # Scalar pentru sunet ( * sunet cu RMS pentru a ajunge la nivelul de normalizare dorit)
    normalized_sound = sound * gain
    return normalized_sound

def resampler(input_dir, speech, target_sr=48000):
    ext = '*.flac' if speech else '*.wav'
    files = glob.glob(f"{input_dir}/" + ext) # pentru ca lucram cu datele amestcate in un folder si facem ulterior distinctia intre speech si noise cu extensia fisierului
    for pathname in files:
        audio, fs = sf.read(pathname)
        audio_resampled = librosa.resample(audio, orig_sr=fs, target_sr=target_sr) # soundfile era obligatoriu doar pentru citit. Librosa are functia direct si mai corect decat am fi facut noi
        sf.write(pathname, audio_resampled, target_sr)


def resize(input_dir, dest_dir, speech, segment_len=10):
    ext = '*.flac' if speech else '*.wav'
    files = glob.glob(f"{input_dir}/" + ext)
    for file_path in files:
        audio, fs = sf.read(file_path)
        required_len = segment_len * fs

        # resize audio depending on difference
        if len(audio) < required_len:
            repeat_times = -(-required_len // len(audio))
            audio = np.tile(audio, repeat_times)[:required_len]
        elif len(audio) > required_len:
            remainder = len(audio) % required_len
            if remainder:
                pad_len = required_len - remainder
                audio = np.concatenate((audio, audio[:pad_len]))

        num_segments = len(audio) // required_len
        audio_segments = np.array_split(audio, num_segments)

        basefilename = os.path.splitext(os.path.basename(file_path))[0]

        for j, segment in enumerate(audio_segments, start=1):
            newname = f"{basefilename}_{j}.wav"  # Always use .wav for output
            destpath = os.path.join(dest_dir, newname)
            sf.write(destpath, segment, fs)

def snr_mixer(speechdir, noisedir, snr_range=(0, 5), target_level=-30):
    speech, sr_speech = librosa.load(speechdir, sr=None)
    noise, sr_noise = librosa.load(noisedir, sr=None)

    speech_normalized = normalize(speech, target_level)
    noise_normalized = normalize(noise, target_level)

    # random signal to noise ratio from range
    snr = np.random.uniform(*snr_range)

    # calculate ratio for desired snr
    speech_power = np.mean(speech_normalized**2)
    noise_power = np.mean(noise_normalized**2)
    required_noise_power = speech_power / (10**(snr / 10))

    # scale noise to get the snr
    noise_gain = np.sqrt(required_noise_power / (noise_power + 1e-10))
    noise_scaled = noise_normalized * noise_gain

    # mix the two and return
    mixed_signal = speech_normalized + noise_scaled

    return mixed_signal

def test():
    test = "Test"
    resampler(test,0)
    resize(test,test,0)
