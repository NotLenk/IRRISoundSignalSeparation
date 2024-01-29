import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_audio(directory):
    all_files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    # Separate files into speech (.flac) and noise (.wav) lists
    speech_files = [os.path.join(directory, file) for file in all_files if file.lower().endswith('.flac')]
    noise_files = [os.path.join(directory, file) for file in all_files if file.lower().endswith('.wav')]

    # Split into training, validation, and test sets
    speech_train, speech_valid_test = train_test_split(speech_files, test_size=0.2, random_state=42)
    speech_valid, speech_test = train_test_split(speech_valid_test, test_size=0.5, random_state=42)

    noise_train, noise_valid_test = train_test_split(noise_files, test_size=0.2, random_state=42)
    noise_valid, noise_test = train_test_split(noise_valid_test, test_size=0.5, random_state=42)

    speech_train_df = pd.DataFrame({
        'index': range(len(speech_train)),
        'filepath': speech_train,
    })

    speech_valid_df = pd.DataFrame({
        'index': range(len(speech_valid)),
        'filepath': speech_valid,
    })

    speech_test_df = pd.DataFrame({
        'index': range(len(speech_test)),
        'filepath': speech_test,
    })

    noise_train_df = pd.DataFrame({
        'index': range(len(noise_train)),
        'filepath': noise_train,
    })

    noise_valid_df = pd.DataFrame({
        'index': range(len(noise_valid)),
        'filepath': noise_valid,
    })

    noise_test_df = pd.DataFrame({
        'index': range(len(noise_test)),
        'filepath': noise_test,
    })

    return speech_train_df, speech_valid_df, speech_test_df, noise_train_df, noise_valid_df, noise_test_df

def createDatasets():
    path = 'Sounds'
    columns = ['index', 'filepath']
    (
        speech_train_df,
        speech_valid_df,
        speech_test_df,
        noise_train_df,
        noise_valid_df,
        noise_test_df
    ) = load_audio(path)

    # Save to CSV files
    speech_train_df.to_csv('Speech_Train.csv', index=False)
    speech_valid_df.to_csv('Speech_Valid.csv', index=False)
    speech_test_df.to_csv('Speech_Test.csv', index=False)

    noise_train_df.to_csv('Noise_Train.csv', index=False)
    noise_valid_df.to_csv('Noise_Valid.csv', index=False)
    noise_test_df.to_csv('Noise_Test.csv', index=False)

createDatasets()
