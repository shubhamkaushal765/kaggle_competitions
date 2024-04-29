import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_exploration import Data
import seaborn as sns
import polars as pl
from sklearn.cluster import KMeans
import joblib

# Example Data
np.random.seed(42)
ds = Data(csv_path=None)

y = ds.df_tr["column_0"].to_list()
X = ds.df_tr.drop("column_0")
X = X[[f"column_{i}" for i in list(range(56, 56 + 140))]]  # choose columns

# Step 1: Standardize the Data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Step 2-5: PCA
def plot_pca():
    y = list(map(int, y))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    X_pca = pl.DataFrame(X_pca, schema=["a", "b"])

    X_pca = X_pca.with_columns(label=pl.Series(y))
    print(X_pca)
    pal = sns.color_palette("hls", len(X_pca["label"].unique()))

    p1 = sns.scatterplot(
        x="a",  # Horizontal axis
        y="b",  # Vertical axis
        data=X_pca,  # Data source
        hue="label",
        legend=False,
        palette=pal,
    )
    plt.show()
    exit()

    # Plot Explained Variance Ratio
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)

    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs. Number of Principal Components")
    plt.show()


pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_std)
# labels = len(np.unique(y))
# print(labels)
# km = KMeans(n_clusters=labels)
# km.fit(X_pca)
# joblib.dump(km, "ckpts/km.sav")
# print(np.unique(km.labels_, return_counts=True))
km = joblib.load("ckpts/km.sav")

for i in np.unique(y):
    X_temp = X_pca[np.array(y) == i]
    results = km.predict(X_temp)
    plt.hist(results)
    plt.show()
    # exit()
