import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_vertex_statistics(stats_summary, mean_v, std_v, skew_v):
    fig = plt.figure(figsize=(20, 10))

    plt.subplot(231)
    data = {"x": mean_v[0], "y": mean_v[1], "z": mean_v[2]}
    sns.boxplot(data=data)
    plt.title("Vertex Distribution per Axis")
    plt.ylabel("Values")

    plt.subplot(232)
    for i, axis in enumerate(["x", "y", "z"]):
        sns.histplot(mean_v[i], label=axis, alpha=0.5)
    plt.title("Vertex Value Distribution")
    plt.legend()

    plt.subplot(233)
    axes = ["x", "y", "z"]
    min_vals = [stats_summary[f"{axis}_axis"]["min"] for axis in axes]
    max_vals = [stats_summary[f"{axis}_axis"]["max"] for axis in axes]
    x = np.arange(len(axes))
    width = 0.35
    plt.bar(x - width / 2, min_vals, width, label="Min")
    plt.bar(x + width / 2, max_vals, width, label="Max")
    plt.xticks(x, axes)
    plt.title("Min/Max Values per Axis")
    plt.legend()

    # 4. Violin plot for standard deviations
    plt.subplot(234)
    std_data = {"x": std_v[0], "y": std_v[1], "z": std_v[2]}
    sns.violinplot(data=std_data)
    plt.title("Standard Deviation Distribution")

    # 5. Scatter plot of mean vs std
    plt.subplot(235)
    for i, axis in enumerate(["x", "y", "z"]):
        plt.scatter(mean_v[i], std_v[i], label=axis, alpha=0.5)
    plt.xlabel("Mean")
    plt.ylabel("Standard Deviation")
    plt.title("Mean vs Standard Deviation")
    plt.legend()

    # 6. Skewness distribution
    plt.subplot(236)
    skew_data = {"x": skew_v[0], "y": skew_v[1], "z": skew_v[2]}
    sns.violinplot(data=skew_data)
    plt.title("Skewness Distribution")

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(mean_v):
    corr_matrix = np.corrcoef([mean_v[0], mean_v[1], mean_v[2]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        xticklabels=["x", "y", "z"],
        yticklabels=["x", "y", "z"],
        cmap="coolwarm",
    )
    plt.title("Correlation Matrix between Axes")
    plt.show()


# plot_vertex_statistics(stats_summary, mean_v, std_v, skew_v)


# plot_correlation_matrix(mean_v)

# To read the data back
loaded_lower_df = pd.read_csv("lower_jaw_statistics.csv")
loaded_upper_df = pd.read_csv("upper_jaw_statistics.csv")
loaded_combined_df = pd.read_csv("combined_jaw_statistics.csv")
loaded_vertex_df = pd.read_csv("vertex_count_statistics.csv")
