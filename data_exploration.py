import os
import librosa
from tqdm import tqdm
import numpy as np
import polars as pl
import matplotlib.pyplot as plt


class Data:
    def __init__(
        self, csv_path="data/birdclef2024/train_metadata.csv", plot_hist=False
    ):
        """
        Initialize the Data class.

        Parameters:
        - csv_path (str): Path to the CSV file containing metadata.
        - plot_hist (bool): If True, plot a histogram of category counts.

        This class is used for data analytics and feature extraction from audio files.
        """

        self.df = pl.read_csv(csv_path)
        self.df = self.df[["primary_label", "rating", "url", "filename"]]

        # Mapping labels to unique identifiers
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
        """
        Generate analytics based on label counts.

        This method calculates and prints various statistics on label counts.
        """

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

    def audio_features(
        self,
        index,
        sampling_duration=5,
        percentile_width=25,
        stft_width=8,
        data_root="./data/birdclef2024/train_audio/",
    ):
        """
        Extract audio features from an audio file.
        Sources:
        - https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d

        Parameters:
        - index (int): Index of the file in the dataframe.
        - sampling_duration (int): Duration in seconds to sample the audio.
        - percentile_width (int): Width of percentile calculation.
        - stft_width (int): Width of Short-Time Fourier Transform.
        - data_root (str): Root directory containing audio files.

        Returns:
        - DataFrame: Extracted audio features as a Polars DataFrame.
        """

        row = self.df[index]
        audio_path = os.path.join(data_root, row["filename"][0])
        label_id = self.name2id[os.path.basename(os.path.dirname(audio_path))]

        x, sr = librosa.load(audio_path, sr=None)
        chunk_size = sampling_duration * sr

        output_df = pl.DataFrame()

        # chunking out sampling duration
        for i in range(0, len(x), chunk_size):

            x_sample = x[i : i + chunk_size]
            x_sample = librosa.util.fix_length(x_sample, size=chunk_size)
            if sum(x_sample) == 0:
                continue

            # Short Term Fourier Transform
            X = np.array(librosa.stft(x_sample))
            Xdb = np.array(librosa.amplitude_to_db(abs(X)))[::stft_width]
            # Zero Crossing Rate
            zerox = librosa.feature.zero_crossing_rate(x_sample)
            # Spectral Centroid & Spectral Rolloff
            s_divisor = 1000
            s_ctrds = librosa.feature.spectral_centroid(y=x_sample, sr=sr) / s_divisor
            s_rolloff = librosa.feature.spectral_rolloff(y=x_sample, sr=sr) / s_divisor
            # Mel-Frequency Cepstral Coefficients
            mfccs = librosa.feature.mfcc(y=x_sample, sr=sr)

            # Generate relevant statistics from metrics
            output = [label_id]
            for m in [Xdb, zerox, s_ctrds, s_rolloff, mfccs]:
                output.extend(m.mean(axis=-1).tolist() + m.std(axis=-1).tolist())
                for p in range(0, 101, percentile_width):
                    output.extend(np.percentile(m, p, axis=-1).tolist())
            # assert len(output) == 1048 * 13 + 1, (len(output), 1048 * 13 + 1)
            output_df = pl.concat([output_df, pl.DataFrame(output).transpose()])
        return output_df


if __name__ == "__main__":
    ds = Data()
    bucket_size = 500
    for df_i in tqdm(range(0, len(ds.df), bucket_size)):
        dfs = []

        for i in tqdm(range(df_i, df_i + bucket_size)):
            dfs.append(ds.audio_features(index=i))

        dfs = pl.concat(dfs)
        dfs.write_csv(f"./data/processed_pickles/_{df_i}_{df_i + bucket_size}.csv")
