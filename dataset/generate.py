import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import pathlib
import sys


DATA_PATH = "data"
GENRE_MAPPING = {
    "test": 0,
    "house": 1,
    "trance": 2,
    "techno": 3,
    "hardstyle": 4,
    "trap": 5,
    "dubstep": 6,
    "electro": 7,
}


def generate_dataset(genre):
    csv_file = f"{DATA_PATH}/output/{genre}.csv"
    genre_path = f"{DATA_PATH}/{genre}/"

    df = (
        pd.read_csv(csv_file)
        if pathlib.Path(csv_file).exists()
        else pd.DataFrame(
            columns=[
                "filename",
                "length",
                "chroma_stft_mean",
                "chroma_stft_var",
                "rms_mean",
                "rms_var",
                "spectral_centroid_mean",
                "spectral_centroid_var",
                "spectral_bandwidth_mean",
                "spectral_bandwidth_var",
                "rolloff_mean",
                "rolloff_var",
                "zero_crossing_rate_mean",
                "zero_crossing_rate_var",
                "harmony_mean",
                "harmony_var",
                "perceptr_mean",
                "perceptr_var",
                "tempo",
                "mfcc1_mean",
                "mfcc1_var",
                "mfcc2_mean",
                "mfcc2_var",
                "mfcc3_mean",
                "mfcc3_var",
                "mfcc4_mean",
                "mfcc4_var",
                "mfcc5_mean",
                "mfcc5_var",
                "mfcc6_mean",
                "mfcc6_var",
                "mfcc7_mean",
                "mfcc7_var",
                "mfcc8_mean",
                "mfcc8_var",
                "mfcc9_mean",
                "mfcc9_var",
                "mfcc10_mean",
                "mfcc10_var",
                "mfcc11_mean",
                "mfcc11_var",
                "mfcc12_mean",
                "mfcc12_var",
                "mfcc13_mean",
                "mfcc13_var",
                "mfcc14_mean",
                "mfcc14_var",
                "mfcc15_mean",
                "mfcc15_var",
                "mfcc16_mean",
                "mfcc16_var",
                "mfcc17_mean",
                "mfcc17_var",
                "mfcc18_mean",
                "mfcc18_var",
                "mfcc19_mean",
                "mfcc19_var",
                "mfcc20_mean",
                "mfcc20_var",
                "genre",
            ],
        )
    )

    files = os.listdir(genre_path)
    exist_filenames = set(df["filename"].tolist())
    pbar = tqdm(enumerate(files), total=len(files))

    for _, file in pbar:
        pbar.set_description(f"File: {file}")
        if file in exist_filenames:
            continue

        attrs = []

        audio_path = f"{genre_path}/{file}"
        y, sr = librosa.load(audio_path)

        length = librosa.get_duration(y=y, sr=sr)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)

        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)

        harmony = librosa.effects.harmonic(y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)

        perceptr = librosa.effects.percussive(y)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_var = np.var(mfccs, axis=1)

        attrs.extend(
            [
                file,
                length,
                chroma_stft_mean,
                chroma_stft_var,
                rms_mean,
                rms_var,
                spectral_centroid_mean,
                spectral_centroid_var,
                spectral_bandwidth_mean,
                spectral_bandwidth_var,
                rolloff_mean,
                rolloff_var,
                zero_crossing_rate_mean,
                zero_crossing_rate_var,
                harmony_mean,
                harmony_var,
                perceptr_mean,
                perceptr_var,
                tempo,
            ]
        )

        for i in range(1, 21):
            attrs.extend([mfcc_mean[i - 1], mfcc_var[i - 1]])

        attrs.append(GENRE_MAPPING[genre])
        df.loc[len(df), :] = np.asarray(attrs, dtype="object")

    df.to_csv(csv_file, index=False)
    print(f"[+] Dataset [{genre}] created.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python generate.py <genre>")
        sys.exit(1)

    genre = sys.argv[-1]
    generate_dataset(genre)
