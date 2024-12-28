import numpy as np

# import utils.general as g
import os
import trimesh
from tqdm import tqdm
from statistics import mode


def load_obj(path):
    mesh = trimesh.load(path)
    return mesh


path = r"D:\projects\cuda11.8\3dmodels\dataset\obj"
folders = os.listdir(path)
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
max_vertices_l = 0
min_vertices_l = float("inf")
max_vertices_u = 0
min_vertices_u = float("inf")


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


def data_analysis(v, j):
    for axis in range(3):
        axis_data = v[:, axis]

        max_v[j, axis] = max(max_v[axis], np.max(axis_data))
        min_v[j, axis] = min(min_v[axis], np.min(axis_data))
        # mean, median, std, var
        mean_v[axis].append(np.mean(axis_data))
        median_v[axis].append(np.median(axis_data))


for i in tqdm(range(len(folders))):
    npath = path + "\\" + folders[i]
    files = os.listdir(npath)
    j = 0
    for f in files:
        mesh = load_obj(npath + "\\" + f)
        v = np.array(mesh.vertices)
        max_vertices = max(max_vertices, v.shape[0])
        min_vertices = min(min_vertices, v.shape[0])
        data_analysis(v)
        break
