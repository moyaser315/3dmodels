import open3d as o3d
import numpy as np
import utils.general as g


def get_colored_mesh(mesh, label_arr):
    palette = (
        np.array(
            [
                [255, 153, 153],
                [153, 76, 0],
                [153, 153, 0],
                [76, 153, 0],
                [0, 153, 153],
                [0, 0, 153],
                [153, 0, 153],
                [153, 0, 76],
                [64, 64, 64],
                [255, 128, 0],
                [153, 153, 0],
                [76, 153, 0],
                [0, 153, 153],
                [0, 0, 153],
                [153, 0, 153],
                [153, 0, 76],
                [64, 64, 64],
            ]
        )
        / 255
    )
    palette[9:] *= 0.4  # Dim colors for higher labels
    label_arr = label_arr.copy()
    label_arr %= palette.shape[0]  # Ensure label indices wrap around
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, color in enumerate(palette):
        label_colors[label_arr == idx] = color
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    return mesh


def visualize_colored_mesh(mesh):
    o3d.visualization.draw_geometries(
        [mesh],
        mesh_show_wireframe=False,
        mesh_show_back_face=True,
        point_show_normal=True,
    )


if __name__ == "__main__":
    mesh_path = r".\dataset\obj\0EAKT1CU\0EAKT1CU_lower.obj"
    label_json_path = r".\dataset\labels\0EJBIPTC\0EJBIPTC_lower.json"

    tri_mesh = g.load_obj(mesh_path)
    print("loaded triemesh")
    o3d_mesh = g.trimesh_to_open3d(tri_mesh)
    print("reached here")
    labels = np.array(g.load_json(label_json_path)["labels"])
    colored_mesh = get_colored_mesh(o3d_mesh, labels)
    visualize_colored_mesh(colored_mesh)
