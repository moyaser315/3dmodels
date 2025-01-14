import os
import numpy as np
from glob import glob
import open3d as o3d
import pointops
import torch
import json


SAVE_PATH = r"D:\projects\cuda11.8\3dmodels\dataset\processed"
OBJ_PATH = r"D:\projects\cuda11.8\3dmodels\dataset\obj"
LABELS_PATH = r"D:\projects\cuda11.8\3dmodels\dataset\labels"
AXIS_MIN = np.array([-53.6129611, -42.70115327, -132.9866643])
AXIS_MAX = np.array([52.01002367, 42.82501054, -76.30622343])


def resample_pcd(pcd_ls, n):
    idx = furth_point(pcd_ls[0][:, :3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    return pcd_resampled_ls


def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()


def furth_point(xyz, npoint):
    if xyz.shape[0] <= npoint:
        raise "small fig"
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(
        xyz,
        torch.tensor([xyz.shape[0]]).cuda().type(torch.int),
        torch.tensor([npoint]).cuda().type(torch.int),
    )
    return torch_to_numpy(idx).reshape(-1)


def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)


def read_txt_obj_ls(path):

    f = open(path, "r")
    vertex_ls = []
    tri_ls = []

    while True:
        line = f.readline().split()
        if not line:
            break
        if line[0] == "v":
            vertex_ls.append(list(map(float, line[1:4])))
        elif line[0] == "f":
            tri_verts_idxes = list(map(str, line[1:4]))
            if "//" in tri_verts_idxes[0]:
                for i in range(len(tri_verts_idxes)):
                    tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
            tri_verts_idxes = list(map(int, tri_verts_idxes))
            tri_ls.append(tri_verts_idxes)
        else:
            continue
    f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls) - 1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)

    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls, norms], axis=1)]
    output.append(mesh)

    return output


os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

stl_path_ls = []
for dir_path in [x[0] for x in os.walk(OBJ_PATH)][1:]:
    stl_path_ls += glob(os.path.join(dir_path, "*.obj"))

json_path_map = {}
for dir_path in [x[0] for x in os.walk(LABELS_PATH)][1:]:
    for json_path in glob(os.path.join(dir_path, "*.json")):
        json_path_map[os.path.basename(json_path).split(".")[0]] = json_path

all_labels = []
for i in range(len(stl_path_ls)):
    print(i, end=" ")
    base_name = os.path.basename(stl_path_ls[i]).split(".")[0]
    loaded_json = load_json(json_path_map[base_name])
    labels = np.array(loaded_json["labels"]).reshape(-1, 1)
    if loaded_json["jaw"] == "lower":
        labels -= 20
    labels[labels // 10 == 1] %= 10
    labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
    labels[labels < 0] = 0

    vertices, org_mesh = read_txt_obj_ls(stl_path_ls[i])

    # Translation Invariance
    vertices[:, :3] -= np.mean(vertices[:, :3], axis=0)
    
    # scale Invariance
    vertices[:, :3] = ((vertices[:, :3] - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 2 - 1

    labeled_vertices = np.concatenate([vertices, labels], axis=1)

    name_id = str(base_name)
    if labeled_vertices.shape[0] > 24000:
        labeled_vertices = resample_pcd([labeled_vertices], 24000)[0]

    np.save(
        os.path.join(SAVE_PATH, f"{name_id}_{loaded_json['jaw']}_sampled_points"),
        labeled_vertices,
    )
