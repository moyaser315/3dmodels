import trimesh
import json
import open3d as o3d
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_obj(path):
    mesh = trimesh.load(path)
    return mesh


def trimesh_to_open3d(tri_mesh):
    mesh = o3d.geometry.TriangleMesh()
    if len(tri_mesh.vertices) == 0:
        raise ValueError("Empty vertices in trimesh")
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    if tri_mesh.faces is None or len(tri_mesh.faces) == 0:
        raise ValueError("No faces found in trimesh")
    faces = np.asarray(tri_mesh.faces, dtype=np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


if __name__ == "__main__":
    js_p = r".\dataset\labels\0EJBIPTC\0EJBIPTC_lower.json"
    obj_p = r".\dataset\obj\0EAKT1CU\0EAKT1CU_lower.obj"
    print(load_json(js_p)["jaw"])
    print(load_obj(obj_p).vertices)
