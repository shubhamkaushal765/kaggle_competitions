from sklearn.ensemble import RandomForestClassifier
from data_exploration import Data
import pickle
import polars as pl
import os


def log_and_save(logs, log_file="logs.csv", save_dir="ckpts"):
    other_ckpts = os.listdir(save_dir)
    new_index = (
        max(
            list(map(lambda x: int(x.split("_")[-1].replace(".pkl", "")), other_ckpts))
            + [0]
        )
        + 1
    )
    new_ckpt = os.path.join(save_dir, f"rfc_clf_{new_index}.pkl")
    logs["ckpt"] = new_ckpt

    with open(new_ckpt, "wb") as f:
        pickle.dump(clf, f)

    logs = pl.DataFrame(logs)
    if os.path.exists(log_file):
        logs = pl.concat([logs, pl.read_csv(log_file)], how="diagonal")
        os.remove(log_file)
    logs.write_csv(log_file)


seed = 42
logs = {
    "n_estimators": 50,
    "max_depth": 32,
    "oob_score": True,
    "random_state": seed,
    "verbose": 3,
    "class_weight": "balanced",
    "max_features": "log2",
    "n_jobs": -1,
}
clf = RandomForestClassifier(**logs)
ds = Data(csv_path=None)

y = ds.df_tr["column_0"]
X = ds.df_tr.drop("column_0")
clf.fit(X, y)
logs["oob_score"] = clf.oob_score_
log_and_save(logs)
