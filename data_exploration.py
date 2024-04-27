import os
import librosa
import sklearn

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


class Data:
    def __init__(
        self, csv_path="data/birdclef2024/train_metadata.csv", plot_hist=False
    ):
        self.df = pl.read_csv(csv_path)
        self.df = self.df[["primary_label", "rating", "url", "filename"]]

        label_col = "primary_label"
        unique_labels = sorted(list(set(self.df[label_col])))
        self.name2id = {name: id for id, name in enumerate(unique_labels)}
        self.id2name = {id: name for name, id in self.name2id.items()}

        self.vc = self.df[label_col].value_counts()
        if plot_hist:
            plt.figure(figsize=(8, 6))
            plt.bar(self.vc[label_col], self.vc["count"], color="skyblue")
            plt.xlabel("Labels")
            plt.ylabel("Count")
            plt.title("Histogram of Category Counts")
            plt.show()

    def generate_analytics(self):

        vc = self.vc

        # Calculate analytics on the value counts
        total_values = len(vc["count"])
        mmin = vc["count"].min()
        mmax = vc["count"].max()
        mean = vc["count"].mean()
        n_mins = sum(map(lambda x: x == mmin, vc["count"]))
        n_maxs = sum(map(lambda x: x == mmax, vc["count"]))

        print(f"Total values: {total_values}")
        print(f"Minimum count: {mmin}")
        print(f"Maximum count: {mmax}")
        print(f"Average count: {mean:.2f}")
        print(f"Number of min values: {n_mins}")
        print(f"Number of max values: {n_maxs}")

    def audio_features(self, data_root="./data/birdclef2024/train_audio/"):
        """https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d"""

        row = self.df[0]
        audio_path = os.path.join(data_root, row["filename"][0])

        x, sr = librosa.load(audio_path, sr=None)

        # Short Term Fourier Transform
        X = np.array(librosa.stft(x))
        Xdb = np.array(librosa.amplitude_to_db(abs(X)))
        print(f"Short Term Fourier Transform: {X.shape}")
        print(f"STFT in db: {Xdb.shape}")
        print("################################")
        # plt.figure(figsize=(14, 5))
        # librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")  # y_axis='log'
        # plt.colorbar()
        # plt.show()
        ################################################################

        # Zero Crossing Rate
        zero_crossings = librosa.zero_crossings(x, pad=True)
        print(f"Zero Crossings Shape: {zero_crossings.shape}")
        print(f"Non-negative places of zero crossing: {sum(zero_crossings)}")
        print(zero_crossings)
        print("################################")
        ################################################################

        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
        print(f"Spectral Centroids Shape: {spectral_centroids.shape}")
        print("################################")

        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)

        # # Normalising the spectral centroid for visualisation
        # def normalize(x, axis=0):
        #     return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # # Plotting the Spectral Centroid along the waveform
        # # librosa.display.waveplot(x, sr=sr, alpha=0.4)
        # plt.plot(t, spectral_centroids, color="r")
        # plt.show()
        ################################################################

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]
        print(f"Spectral Rolloff Shape: {spectral_rolloff.shape}")
        print("################################")

        # librosa.display.waveshow(y=x, sr=sr)
        # plt.plot(t, spectral_rolloff, color="r")
        # plt.show()
        ################################################################

        # Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=x, sr=sr)
        print(f"Mel-Frequency Cepstral Coefficients: {mfccs.shape}")
        print("################################")
        # # Displaying  the MFCCs:
        # librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        # plt.show()
        ################################################################


if __name__ == "__main__":
    ds = Data()
    # ds.generate_analytics()
    ds.audio_features()
