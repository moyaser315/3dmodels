import json
import os
from glob import glob
import argparse
import numpy as np
import torch
import utils.general as u
from sklearn.neighbors import KDTree
import open3d as o3d
from models.pointnet import PointNet
import sys
sys.path.append(os.getcwd())


class Infer:
    def __init__(self, model):
        self.model = model
        self.scaler = 1.9
        self.shifter = 0.9

    def __call__(self, stl_path):
        _, mesh = u.read_files(stl_path, ret_mesh=True, tri=True)
        vertices = np.array(mesh.vertices)
        
        # Normalize vertices
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        org_feats = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        # Subdivide mesh if needed
        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        sampled_feats = u.resample_pcd([vertices.copy()], 24000)[0]
        
        # Predict classification
        with torch.no_grad():
            input_cuda_feats = torch.from_numpy(np.array([sampled_feats.astype('float32')])).cuda().permute(0,2,1)
            cls_pred = self.model([input_cuda_feats])['cls_pred']
        
        cls_pred = cls_pred.argmax(axis=1)
        cls_pred = u.torch_to_numpy(cls_pred)
        cls_pred[cls_pred>=9] += 2
        cls_pred[cls_pred>0] += 10
        
        # Map predictions to original points
        tree = KDTree(sampled_feats[:,:3], leaf_size=2)
        near_points = tree.query(org_feats[:,:3], k=1, return_distance=False)
        result_labels = cls_pred.reshape(-1)[near_points.reshape(-1)].reshape(-1)

        return result_labels

class Segment:
    def __init__(self, ckpt_path):
        module = PointNet({})
        module.load_state_dict(torch.load(ckpt_path))
        module.cuda()
        self.process_pipeline = Infer(module)

    @staticmethod
    def get_jaw(scan_path):
        filename = os.path.basename(scan_path)
        
        parts = filename.split('.')[0].split('_')
        if len(parts) > 1 and parts[1] in ["upper", "lower"]:
            return parts[1]

        with open(scan_path, 'r') as f:
            jaw = f.readline().strip()[2:]
            if jaw in ["upper", "lower"]:
                return jaw
        
        return None

    def predict(self, inputs):
        if len(inputs) != 1:
            raise ValueError(f"inputs {len(inputs)}")
        
        scan_path = inputs[0]
        labels = self.process_pipeline(scan_path)
        
        jaw = self.get_jaw(scan_path)
        if jaw is None:
            raise ValueError("Invalid jaw name")
        
        if jaw == "lower":
            labels[labels>0] += 20
        
        print("jaw processed is:", jaw)
        labels = labels.astype(int).tolist()

        return labels, jaw


    def process(self, input_path, output_path):
        labels, jaw = self.predict([input_path])
        
        pred_output = {
            'id_patient': "",
            'jaw': jaw,
            'labels': labels
        }

        with open(output_path, 'w') as fp:
            json.dump(pred_output, fp, default=u.json_numpy_default)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training models")
    parser.add_argument(
        "--input_path",
        default="obj",
        type=str,
        help="input data dir path.",
    )
    parser.add_argument(
        "--test_txt", default="test.txt", type=str, help="split txt file path."
    )
    parser.add_argument(
        "--save_path",
        default="result",
        type=str,
        help="input data dir path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="chkpoints/pointnet.pt",
        type=str,
        help="input data dir path.",
    )
    args = parser.parse_args()
    input_p = args.input_path
    test_files = args.test_txt
    save_path = args.save_path
    chk_point = args.checkpoint
    file_names = []
    if test_files != "":
        with open(test_files, 'r') as f:
            file_names = [line.strip() for line in f]

    obj_files = []
    for dir_path in [x[0] for x in os.walk(input_p)][1:]:
        if os.path.basename(dir_path) in file_names: 
            obj_files += glob.glob(os.path.join(dir_path,"*.obj"))

    pred_obj = Segment(chk_point)
    os.makedirs(save_path, exist_ok=True)
    for i, obj_file in enumerate(obj_files):
        print(f"Processing: {i}: {obj_file}")
        pred_obj.process(obj_file, os.path.join(save_path, os.path.basename(obj_file).replace(".obj", ".json")))