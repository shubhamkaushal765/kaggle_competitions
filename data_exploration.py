import polars as pl
import matplotlib.pyplot as plt


class Data:
    def __init__(
        self, csv_path="data/birdclef2024/train_metadata.csv", plot_hist=False
    ):
        self.df = pl.read_csv(csv_path)

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


if __name__ == "__main__":
    ds = Data()
    ds.generate_analytics()
