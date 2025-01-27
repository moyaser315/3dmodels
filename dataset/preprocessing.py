import os
import numpy as np
from glob import glob
from ..utils import general as u
import argparse


parser = argparse.ArgumentParser(description="preprocessing")
parser.add_argument(
    "--obj_path", default="data_preprocessed", type=str, help="save data dir path."
)
parser.add_argument(
    "--save_path", default="data_preprocessed", type=str, help="save data dir path."
)
parser.add_argument("--labels", default="labels", type=str, help="labels")
args = parser.parse_args()
SAVE_PATH = args.save_path
OBJ_PATH = args.obj_path
LABELS_PATH = args.labels
AXIS_MIN = np.array([-53.6129611, -42.70115327, -132.9866643])
AXIS_MAX = np.array([52.01002367, 42.82501054, -76.30622343])

os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

mesh_files = []
for dir_path in [x[0] for x in os.walk(OBJ_PATH)][1:]:
    mesh_files += glob(os.path.join(dir_path, "*.obj"))

json_files = {}
for dir_path in [x[0] for x in os.walk(LABELS_PATH)][1:]:
    for json_file in glob(os.path.join(dir_path, "*.json")):
        json_files[os.path.basename(json_file).split(".")[0]] = json_file

all_labels = []
for i in range(len(mesh_files)):
    print(i, end=" ")
    base_name = os.path.basename(mesh_files[i]).split(".")[0]
    loaded_json = u.load_json(json_files[base_name])
    labels = np.array(loaded_json["labels"]).reshape(-1, 1)
    if loaded_json["jaw"] == "lower":
        labels -= 20
    labels[labels // 10 == 1] %= 10
    labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
    labels[labels < 0] = 0

    vertices, org_mesh = u.read_files(mesh_files[i])

    # Translation Invariance
    vertices[:, :3] -= np.mean(vertices[:, :3], axis=0)

    # scale Invariance
    vertices[:, :3] = ((vertices[:, :3] - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 2 - 1

    labeled_vertices = np.concatenate([vertices, labels], axis=1)

    name_id = str(base_name)
    if labeled_vertices.shape[0] > 24000:
        labeled_vertices = u.resample_pcd([labeled_vertices], 24000)[0]

    np.save(
        os.path.join(SAVE_PATH, f"{name_id}_{loaded_json['jaw']}_sampled_points"),
        labeled_vertices,
    )
