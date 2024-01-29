import os
import librosa
import numpy as np
import glob
import soundfile as sf


EPS = np.finfo(float).eps
np.random.seed(0)

def audioread(path, start=0, stop=None, target_level=-30):
    print("Reading audio at {}".format(path))
    path = os.path.abspath(path)
    audio, sample_rate = sf.read(path, start=start, stop=stop)

    if len(audio.shape) == 1:  # mono
        rms = (audio ** 2).mean() ** 0.5
        scalar = 10 ** (target_level / 20) / (rms + EPS)
        audio = audio * scalar

    return audio, sample_rate

def audiowrite(destpath, audio, sample_rate=48000, target_level=-30):
    print("Writing at {}".format(destpath))
    audio = normalize(audio, target_level)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def normalize(audio, target_level):
    print("Normalizing")
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def resampler(input_dir,speech, target_sr=48000):
    if speech:
        ext = '*.flac'
    else:
        ext = '*.wav'
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print("Resampling {} ".format(pathname) + "Speech {}".format(speech))
        audio, fs = audioread(pathname)
        audio_resampled = librosa.resample(audio, orig_sr=fs, target_sr=48000)
        audiowrite(pathname, audio_resampled, target_sr)


def audio_length_resize(input_dir, dest_dir, speech, segment_len=10):
    if speech:
        ext = '*.flac'
    else:
        ext = '*.wav'
    files = glob.glob(f"{input_dir}/" + ext)
    for i in range(len(files)):
        audio, fs = audioread(files[i])
        print("Relengthening {} ".format(files[i]) + "")
        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(audio, audio[0: segment_len * fs - (len(audio) % (segment_len * fs))])
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)
            audio = audio[:segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + '_' + str(j) + ext
            destpath = os.path.join(dest_dir, newname)
            audiowrite(destpath, audio_segments[j], fs)


def snr_mixer(params, clean, noise, snr, target_level=-30):
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -30 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    return clean, noisenewlevel, noisyspeech, noisy_rms_level

def test():
    resampler('Noise',False)
    audio_length_resize('Noise', 'ProcNoise', False)
    resampler('Speech', True)
    audio_length_resize('Speech', 'ProcSpeech', True)

test()