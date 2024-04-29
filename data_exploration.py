import os
import librosa
from tqdm import tqdm
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def check_for_existing_df(path):
    df = pl.DataFrame()
    if os.path.exists(path):
        print(f"{path} already exists.")
        df = pl.read_csv(path)
        os.remove(path)
    return df


class Data:
    def __init__(
        self,
        csv_path="data/birdclef2024/train_metadata.csv",
        plot_hist=False,
        train_test_path="data/train_val/",
        load_val=False,
        allowed_labels=None,
    ):
        """
        Initialize the Data class.

        Parameters:
        - csv_path (str): Path to the CSV file containing metadata.
        - plot_hist (bool): If True, plot a histogram of category counts.

        This class is used for data analytics and feature extraction from audio files.
        """

        assert (csv_path is None) or (train_test_path is None)

        self.df_tr = pl.DataFrame()
        self.df_val = pl.DataFrame()

        if train_test_path is not None:
            files = os.listdir(train_test_path)
            for f in files:
                if "train" in f:
                    df = pl.read_csv(os.path.join(train_test_path, f))
                    self.df_tr = pl.concat([self.df_tr, df])
                elif load_val and "val" in f:
                    df = pl.read_csv(os.path.join(train_test_path, f))
                    self.df_val = pl.concat([self.df_val, df])
            if allowed_labels:
                tr_temp, val_temp = [], []
                for l in allowed_labels:
                    tr_temp.append([self.df_tr.filter(self.df_tr["column_0"] == l)])
                    val_temp.append([self.df_val.filter(self.df_val["column_0"] == l)])
                self.df_tr = pl.concat(tr_temp)
                self.df_val = pl.concat(val_temp)
        else:
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
        stft_width=256,
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

    def train_test_split(self, seed=42, processed_pickles="./data/processed_pickles/"):
        pickles = os.listdir(processed_pickles)
        train_val_path = "./data/train_val/"

        for pkl in tqdm(pickles):
            file_path = os.path.join(processed_pickles, pkl)
            df = pl.read_csv(file_path)
            labels = map(int, df["column_0"].unique().to_list())
            for l in labels:

                # check for existing df
                tr_path = os.path.join(train_val_path, f"train_{l}.csv")
                df_tr = check_for_existing_df(tr_path)
                val_path = os.path.join(train_val_path, f"val_{l}.csv")
                df_val = check_for_existing_df(val_path)

                # train test split
                df_l = df.filter(pl.col("column_0") == l)
                val_size = int(np.ceil(len(df_l) * 0.25))
                X_tr, X_val = train_test_split(
                    df_l, test_size=val_size, random_state=seed
                )

                # appending and saving df
                df_tr = pl.concat([df_tr, X_tr])
                df_val = pl.concat([df_val, X_val])
                df_tr.write_csv(tr_path)
                df_val.write_csv(val_path)

                self.df_tr = pl.concat([self.df_tr, X_tr])
                self.df_val = pl.concat([self.df_val, X_val])


if __name__ == "__main__":
    ds = Data(csv_path=None)

    # STEP 1:
    # Generate Processed Pickles
    # bucket_size = 2000
    # for df_i in tqdm(range(24000, len(ds.df), bucket_size)):
    #     dfs = []

    #     for i in tqdm(range(df_i, df_i + bucket_size)):
    #         try:
    #             dfs.append(ds.audio_features(index=i))
    #         except:
    #             continue
    #     dfs = pl.concat(dfs)
    #     dfs.write_csv(f"./data/processed_pickles/_{df_i}_{df_i + bucket_size}.csv")

    # STEP 2:
    # ds.train_test_split()

    # STEP 3: train data exploration
    # cols: {0: label_id, (1-5)*7: Xdb, 6*7: zerox, 7*7: spectral centroid, 8*7: spectral rolloff, (9-28)*7: mfcc}
    ds.explore_train_data()
