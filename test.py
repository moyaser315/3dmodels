import numpy as np
import torch
import utils.general as utils
from sklearn.neighbors import KDTree
import open3d as o3d



class Infer:
    def __init__(self, model):
        self.model = model
        self.scaler = 2
        self.shifter = 1

    def __call__(self, stl_path):
        _, mesh = utils.read_txt_obj_ls(stl_path, ret_mesh=True, use_tri_mesh=True)
        vertices = np.array(mesh.vertices)
        
        #invariance
        vertices[:,:3] -= np.mean(vertices[:,:3], axis=0)
        vertices[:, :3] = ((vertices[:, :3]-np.min(vertices[:,1]))/(np.max(vertices[:,1])- np.min(vertices[:,1])))*self.scaler-self.shifter
        
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        point_feat = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))

        #handle less than sampling
        if np.asarray(mesh.vertices).shape[0] < 24000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        
        vertices = np.array(np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals)], axis=1))
        sampled_mesh = utils.resample_pcd([vertices.copy()], 24000, "fps")[0]
        
        with torch.no_grad():
            input_cuda_feats = torch.from_numpy(np.array([sampled_mesh.astype('float32')])).cuda().permute(0,2,1)
            cls_pred = self.model([input_cuda_feats])['cls_pred']
        
        cls_pred = cls_pred.argmax(axis=1)
        cls_pred = utils.torch_to_numpy(cls_pred)
        cls_pred[cls_pred>=9] += 2
        cls_pred[cls_pred>0] += 10
        
        tree = KDTree(sampled_mesh[:,:3], leaf_size=2)
        near_points = tree.query(point_feat[:,:3], k=1, return_distance=False)
        result_ins_labels = cls_pred.reshape(-1)[near_points.reshape(-1)].reshape(-1,1)

        return {
            "sem": result_ins_labels.reshape(-1),
            "ins": result_ins_labels.reshape(-1),
        }