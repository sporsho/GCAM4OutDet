import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import yaml
import open3d.visualization as vis

def convert_to_color(color_map, labels):
    rgb = list()
    for l in labels:
        rgb.append(np.array(color_map[l]))
    return np.stack(rgb)

if __name__ == "__main__":
    conf_path = '../3D_OutDet/binary_desnow_wads.yaml'
    # conf_path = 'semanticstf.yaml'
    with open(conf_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    color_map = semkittiyaml['color_map']
    labels = semkittiyaml['labels']
    learning_map = semkittiyaml['learning_map']
    inv_learning_map = semkittiyaml['learning_map_inv']
    seq_root = '/var/local/home/aburai/DATA/WADS2/sequences'
    # lab_root = seq_root
    lab_root = "/var/local/home/aburai/DATA/exp_2023/bin_seg/KDTreeV2/outputs/sequences"

    grad_file = '/var/local/home/aburai/DATA/3D_OutDet/bin_desnow_wads/outputs/sequences/12/grads/039603.grad'
    pc_file = '/var/local/home/aburai/DATA/WADS2/sequences/12/velodyne/039603.bin'
    pred_file = '/var/local/home/aburai/DATA/exp_2023/bin_seg/KDTreeV2/outputs/sequences/12/predictions/039603.label'


    pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
    grad = np.fromfile(grad_file, dtype=np.float32).reshape(-1)
    pred = np.fromfile(pred_file, np.int32).reshape(-1)
    pred = convert_to_color(color_map, pred)
    # pred[:, 2] = grad
    # pred[:, 1] = grad
    # pred[:, 0] = grad
    # pred = pred * 255
    ind = np.nonzero(grad)[0]
    normals = np.vstack((grad, np.zeros_like(grad), np.zeros_like(grad))).swapaxes(0,1)
    mesh_pt = pc[ind]
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_pt[:, : 3])
    mesh_pcd.colors = o3d.utility.Vector3dVector(pred[ind])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(mesh_pcd, 3)
    # radii = [0.005, 0.01, 0.02, 0.04]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(mesh_pcd, o3d.utility.DoubleVector(radii))
    grad_col = plt.get_cmap("plasma")((grad - grad.min()) / (grad.max() - grad.min()))
    grad_col = grad_col[:, :3]
    grad_col[:, 2] = grad
    mesh.compute_vertex_normals()
    dmesh = o3d.geometry.TriangleMesh()
    dmesh.vertices = mesh.vertices
    dmesh.triangles = mesh.triangles
    dmesh.triangle_normals = mesh.triangle_normals
    dmesh.vertex_colors = o3d.utility.Vector3dVector(grad_col[ind])

    # mesh.vertex_colors = o3d.utility.Vector3dVector(pred[ind])
    mat_mesh = o3d.visualization.rendering.MaterialRecord()
    mat_mesh.shader = 'defaultLit'
    mat_mesh.base_color = [0.47, 0.47, 0.45, 0.2]
    mat_mesh.base_roughness = 0.0
    mat_mesh.base_reflectance = 0.0
    mat_mesh.base_clearcoat = 1.0
    mat_mesh.thickness = 1.0
    mat_mesh.transmission = 1.0
    mat_mesh.absorption_distance = 10.0
    mat_mesh.absorption_color = [0.5, 0.5, 0.5]
    # mesh.paint_uniform_color([0.1, 0.7, 0.1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, : 3])
    pcd.colors = o3d.utility.Vector3dVector(pred)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 0.1
    vis.draw([{'name': 'pcd', 'geometry': pcd, 'material':mat},
              {'name': 'box', 'geometry': dmesh, 'material': mat_mesh}], show_skybox=False)
