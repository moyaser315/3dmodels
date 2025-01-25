import trimesh
import json
import open3d as o3d
import numpy as np


# loading files
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_obj(path):
    mesh = trimesh.load_mesh(path, process=False)
    vertex = np.array(mesh.vertices)
    faces = np.array(mesh.faces) + 1
    return vertex, faces


def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()


def json_numpy_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def trimesh_to_open3d(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    if len(vertices) == 0:
        raise ValueError("Empty vertices in trimesh")
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if faces is None or len(faces) == 0:
        raise ValueError("No faces found in trimesh")
    face = np.array(faces) - 1
    mesh.triangles = o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()

    return mesh


def resample_pcd(pcd_ls, n):

    idx = furth_point(pcd_ls[0][:, :3], n)
    pcd_resampled = []
    for i in range(len(pcd_ls)):
        pcd_resampled.append(pcd_ls[i][idx[:n]])
    return pcd_resampled


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


def read_files(path, ret_mesh=False, tri=False):

    if tri:
        vertex, faces = load_obj(path)

    else:
        f = open(path, "r")
        vertex = []
        faces = []

        while True:
            line = f.readline().split()
            if not line:
                break
            if line[0] == "v":
                vertex.append(list(map(float, line[1:4])))

            elif line[0] == "f":
                vertex_idx = list(map(str, line[1:4]))
                if "//" in vertex_idx[0]:
                    for i in range(len(vertex_idx)):
                        vertex_idx[i] = vertex_idx[i].split("//")[0]
                vertex_idx = list(map(int, vertex_idx))
                faces.append(vertex_idx)
            else:
                continue
        f.close()

    mesh = trimesh_to_open3d(vertex, faces)

    norms = np.array(mesh.vertex_normals)
    vertex = np.array(vertex)
    output = [np.concatenate([vertex, norms], axis=1)]

    if ret_mesh:
        output.append(mesh)
    return output


if __name__ == "__main__":
    js_p = r".\dataset\labels\0EJBIPTC\0EJBIPTC_lower.json"
    obj_p = r".\dataset\obj\0EAKT1CU\0EAKT1CU_lower.obj"
    print(load_json(js_p)["jaw"])
    print(load_obj(obj_p).vertices)
