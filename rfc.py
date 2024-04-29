from sklearn.ensemble import RandomForestClassifier
from data_exploration import Data
import pickle
import polars as pl
import os, joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from glob import glob


class RFC:

    def __init__(self, allowed_labels=None):
        self.seed = 42
        self.logs = {
            "n_estimators": 50,
            "max_depth": 32,
            "oob_score": True,
            "random_state": self.seed,
            "verbose": 1,
            "class_weight": "balanced",
            "max_features": "log2",
            "n_jobs": 7,
        }
        self.allowed_labels = allowed_labels

    def log_and_save(self, clf, logs, log_file="logs_1v1.csv", save_dir="ckpts"):
        other_ckpts = glob(os.path.join(save_dir, "rfc*.sav"))
        new_index = (
            max(
                list(
                    map(
                        lambda x: int(x.split("_")[-1].replace(".sav", "")), other_ckpts
                    )
                )
                + [0]
            )
            + 1
        )
        new_ckpt = os.path.join(save_dir, f"rfc_clf_{new_index}.sav")
        logs["ckpt"] = new_ckpt
        joblib.dump(clf, new_ckpt)

        logs = pl.DataFrame(logs)
        if os.path.exists(log_file):
            logs = pl.concat([logs, pl.read_csv(log_file)], how="diagonal")
            os.remove(log_file)
        logs.write_csv(log_file)

    def train_and_save(self, comment, col_start=56, col_end=56 + 140):

        logs = self.logs
        clf = RandomForestClassifier(**logs)
        ds = Data(csv_path=None, allowed_labels=self.allowed_labels)

        y = ds.df_tr["column_0"]
        print(f"Allowed labels: {self.allowed_labels}")
        print(f"Labels found: {list(map(int, y.unique().to_list()))}")

        X = ds.df_tr.drop("column_0")
        X = X[
            [f"column_{i}" for i in list(range(col_start, col_end))]
        ]  # choose columns = mfcc

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_std)

        clf.fit(X, y)
        logs["oob_score"] = clf.oob_score_
        # logs["data"] = "s-centroid-43-50"
        # logs["data"] = "s-rolloff-50-57"
        # logs["data"] = "s-scaler-pca50-mfcc"
        # logs["data"] = "stft-1-36"
        # logs["data"] = "zerox-36-43"
        logs["data"] = comment

        self.log_and_save(clf, logs)


if __name__ == "__main__":
    for i in range(30, 190, 10):
        allowed_labels = list(range(i, i + 10))
        rfc = RFC(allowed_labels=allowed_labels)
        rfc.train_and_save(comment=f"scaler-mfcc-labels-{i}-{i+10}")
