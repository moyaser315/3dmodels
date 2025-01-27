import numpy as np
import os
import trimesh
from tqdm import tqdm
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############ constants
# 0 lower, 1 upper
min_v = [[], []]
max_v = [[], []]
mean_v = [[], []]
median_v = [[], []]
std_v = [[], []]
var_v = [[], []]
q1_v = [[], []]
q3_v = [[], []]
skew_v = [[], []]
kurt_v = [[], []]
max_vertices = 0
min_vertices = float("inf")
stat_combined = {}
stat_combined["all"] = {}
stats_summary = {}


def load_obj(path):
    mesh = trimesh.load(path)
    return mesh


# visualize
def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_vertex_statistics(stats_summary, t, j, mean_v, std_v, skew_v):
    # 1. Boxplot for vertex distribution
    fig = plt.figure(figsize=(8, 6))
    data = {"x": mean_v[j][0], "y": mean_v[j][1], "z": mean_v[j][2]}
    sns.boxplot(data=data)
    plt.title("Vertex Distribution per Axis")
    plt.ylabel("Values")
    save_figure(fig, f"{t}_vertex_distribution.png")

    # 2. Histogram for vertex value distribution
    fig = plt.figure(figsize=(8, 6))
    for i, axis in enumerate(["x", "y", "z"]):
        sns.histplot(mean_v[j][i], label=axis, alpha=0.5)
    plt.title("Vertex Value Distribution")
    plt.legend()
    save_figure(fig, f"{t}_vertex_value_distribution.png")

    # 3. Bar plot for Min/Max values
    fig = plt.figure(figsize=(8, 6))
    axes = ["x", "y", "z"]
    min_vals = [stats_summary[t][f"{axis}_axis"]["min"] for axis in axes]
    max_vals = [stats_summary[t][f"{axis}_axis"]["max"] for axis in axes]
    x = np.arange(len(axes))
    width = 0.35
    plt.bar(x - width / 2, min_vals, width, label="Min")
    plt.bar(x + width / 2, max_vals, width, label="Max")
    plt.xticks(x, axes)
    plt.title("Min/Max Values per Axis")
    plt.legend()
    save_figure(fig, f"{t}_min_max_values.png")

    # 4. Violin plot for standard deviations
    fig = plt.figure(figsize=(8, 6))
    std_data = {"x": std_v[j][0], "y": std_v[j][1], "z": std_v[j][2]}
    sns.violinplot(data=std_data)
    plt.title("Standard Deviation Distribution")
    save_figure(fig, f"{t}_std_deviation_distribution.png")

    # 5. Scatter plot of mean vs std
    fig = plt.figure(figsize=(8, 6))
    for i, axis in enumerate(["x", "y", "z"]):
        plt.scatter(mean_v[j][i], std_v[j][i], label=axis, alpha=0.5)
    plt.xlabel("Mean")
    plt.ylabel("Standard Deviation")
    plt.title("Mean vs Standard Deviation")
    plt.legend()
    save_figure(fig, f"{t}_mean_vs_std.png")

    # 6. Violin plot for skewness distribution
    fig = plt.figure(figsize=(8, 6))
    skew_data = {"x": skew_v[j][0], "y": skew_v[j][1], "z": skew_v[j][2]}
    sns.violinplot(data=skew_data)
    plt.title("Skewness Distribution")
    save_figure(fig, f"{t}_skewness_distribution.png")


def plot_correlation_matrix(mean_v, j):
    corr_matrix = np.corrcoef([mean_v[j][0], mean_v[j][1], mean_v[j][2]])
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        xticklabels=["x", "y", "z"],
        yticklabels=["x", "y", "z"],
        cmap="coolwarm",
    )
    plt.title("Correlation Matrix between Axes")
    save_figure(fig, f'{"upper" if j==1 else "lower"}_correlation_matrix.png')


def init_var():
    for jaw in range(2):
        for i in range(3):
            min_v[jaw].append(float("inf"))
            max_v[jaw].append(float("-inf"))
            mean_v[jaw].append([])
            median_v[jaw].append([])
            std_v[jaw].append([])
            var_v[jaw].append([])
            q1_v[jaw].append([])
            q3_v[jaw].append([])
            skew_v[jaw].append([])
            kurt_v[jaw].append([])


def get_values(v, j):
    for axis in range(3):
        axis_data = v[:, axis]

        max_v[j][axis] = max(max_v[j][axis], np.max(axis_data))
        min_v[j][axis] = min(min_v[j][axis], np.min(axis_data))
        # mean, median, std, var
        mean_v[j][axis].append(np.mean(axis_data))
        median_v[j][axis].append(np.median(axis_data))
        std_v[j][axis].append(np.std(axis_data))
        var_v[j][axis].append(np.var(axis_data))

        # Quartiles
        q1_v[j][axis].append(np.percentile(axis_data, 25))
        q3_v[j][axis].append(np.percentile(axis_data, 75))

        # stats
        skew_v[j][axis].append(stats.skew(axis_data))
        kurt_v[j][axis].append(stats.kurtosis(axis_data))


def get_stats(folders, path):
    j = 0
    for i in tqdm(range(len(folders))):
        npath = path + "\\" + folders[i]
        files = os.listdir(npath)
        j %= 2
        for f in files:
            mesh = load_obj(npath + "\\" + f)
            v = np.array(mesh.vertices)
            max_vertices = max(max_vertices, v.shape[0])
            min_vertices = min(min_vertices, v.shape[0])
            get_values(v, j)
            j += 1


def data_analysis():
    stats_summary = {
        "vertex_count": {
            "min": min_vertices,
            "max": max_vertices,
            "range": max_vertices - min_vertices,
        }
    }
    for i in range(2):
        jaw = ["lower", "upper"][i]
        stats_summary[jaw] = {}
        for axis in range(3):
            axis_name = ["x", "y", "z"][axis]
            stats_summary[jaw][f"{axis_name}_axis"] = {
                "min": min_v[i][axis],
                "max": max_v[i][axis],
                "range": max_v[i][axis] - min_v[i][axis],
                "mean": np.mean(mean_v[i][axis]),
                "median": np.mean(median_v[i][axis]),
                "std": np.mean(std_v[i][axis]),
                "var": np.mean(var_v[i][axis]),
                "q1": np.mean(q1_v[i][axis]),
                "q3": np.mean(q3_v[i][axis]),
                "iqr": np.mean(q3_v[i][axis]) - np.mean(q1_v[i][axis]),
                "skewness": np.mean(skew_v[i][axis]),
                "kurtosis": np.mean(kurt_v[i][axis]),
            }


def combine_stats():
    for axis in range(3):
        axis_name = ["x", "y", "z"][axis]
        mi = min(
            stats_summary["upper"][f"{axis_name}_axis"]["min"],
            stats_summary["lower"][f"{axis_name}_axis"]["min"],
        )
        ma = max(
            stats_summary["upper"][f"{axis_name}_axis"]["max"],
            stats_summary["lower"][f"{axis_name}_axis"]["max"],
        )
        q1 = np.mean(
            [
                stats_summary["upper"][f"{axis_name}_axis"]["q1"],
                stats_summary["lower"][f"{axis_name}_axis"]["q1"],
            ]
        )
        q3 = np.mean(
            [
                stats_summary["upper"][f"{axis_name}_axis"]["q3"],
                stats_summary["lower"][f"{axis_name}_axis"]["q3"],
            ]
        )
        stat_combined["all"][f"{axis_name}_axis"] = {
            "min": mi,
            "max": ma,
            "range": ma - mi,
            "mean": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["mean"],
                    stats_summary["lower"][f"{axis_name}_axis"]["mean"],
                ]
            ),
            "median": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["median"],
                    stats_summary["lower"][f"{axis_name}_axis"]["median"],
                ]
            ),
            "std": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["std"],
                    stats_summary["lower"][f"{axis_name}_axis"]["std"],
                ]
            ),
            "var": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["var"],
                    stats_summary["lower"][f"{axis_name}_axis"]["var"],
                ]
            ),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skewness": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["skewness"],
                    stats_summary["lower"][f"{axis_name}_axis"]["skewness"],
                ]
            ),
            "kurtosis": np.mean(
                [
                    stats_summary["upper"][f"{axis_name}_axis"]["kurtosis"],
                    stats_summary["lower"][f"{axis_name}_axis"]["kurtosis"],
                ]
            ),
        }


def create_jaw_df(stats_summary, jaw_name):
    data = {
        "axis": ["x", "y", "z"],
        "min": [
            stats_summary[jaw_name][f"{axis}_axis"]["min"] for axis in ["x", "y", "z"]
        ],
        "max": [
            stats_summary[jaw_name][f"{axis}_axis"]["max"] for axis in ["x", "y", "z"]
        ],
        "range": [
            stats_summary[jaw_name][f"{axis}_axis"]["range"] for axis in ["x", "y", "z"]
        ],
        "mean": [
            stats_summary[jaw_name][f"{axis}_axis"]["mean"] for axis in ["x", "y", "z"]
        ],
        "median": [
            stats_summary[jaw_name][f"{axis}_axis"]["median"]
            for axis in ["x", "y", "z"]
        ],
        "std": [
            stats_summary[jaw_name][f"{axis}_axis"]["std"] for axis in ["x", "y", "z"]
        ],
        "var": [
            stats_summary[jaw_name][f"{axis}_axis"]["var"] for axis in ["x", "y", "z"]
        ],
        "q1": [
            stats_summary[jaw_name][f"{axis}_axis"]["q1"] for axis in ["x", "y", "z"]
        ],
        "q3": [
            stats_summary[jaw_name][f"{axis}_axis"]["q3"] for axis in ["x", "y", "z"]
        ],
        "iqr": [
            stats_summary[jaw_name][f"{axis}_axis"]["iqr"] for axis in ["x", "y", "z"]
        ],
        "skewness": [
            stats_summary[jaw_name][f"{axis}_axis"]["skewness"]
            for axis in ["x", "y", "z"]
        ],
        "kurtosis": [
            stats_summary[jaw_name][f"{axis}_axis"]["kurtosis"]
            for axis in ["x", "y", "z"]
        ],
    }
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    path = r"D:\projects\cuda11.8\3dmodels\dataset\obj"
    folders = os.listdir(path)
    init_var()
    get_stats(folders, path)
    data_analysis()
    combine_stats()
    lower_df = create_jaw_df(stats_summary, "lower")
    upper_df = create_jaw_df(stats_summary, "upper")
    all_df = create_jaw_df(stat_combined, "all")

    lower_df.to_csv("lower_jaw_statistics.csv", index=False)
    upper_df.to_csv("upper_jaw_statistics.csv", index=False)
    all_df.to_csv("statistics.csv", index=False)

    vertex_df = pd.DataFrame(
        {
            "metric": ["min_vertices", "max_vertices", "range"],
            "value": [
                stats_summary["vertex_count"]["min"],
                stats_summary["vertex_count"]["max"],
                stats_summary["vertex_count"]["range"],
            ],
        }
    )
    vertex_df.to_csv("vertex_count_statistics.csv", index=False)
    plot_vertex_statistics(stats_summary, "lower", 0, mean_v, std_v, skew_v)
    plot_vertex_statistics(stats_summary, "upper", 1, mean_v, std_v, skew_v)
    plot_correlation_matrix(mean_v, 0)
    plot_correlation_matrix(mean_v, 1)
