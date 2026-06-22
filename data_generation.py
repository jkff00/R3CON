import hydra
import torch
import time
import warnings
import torch.multiprocessing as mp
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import repeat

from utils.common import Mapper2Gui, Camera
from utils.operations import random_rotation
from visualization import gui
from simulator import get_simulator
from planning import get_planner
from mapping.voxel_map import VoxelMap
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import math 
warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def downsample_irregular_points(points, sample_interval=[1,1,0.75]):

    dx, dy, dz = sample_interval
    

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    

    x_bins = np.floor(x_coords / dx) * dx 
    y_bins = np.floor(y_coords / dy) * dy 
    z_bins = np.floor(z_coords / dz) * dz 

 
    bins = list(zip(x_bins, y_bins, z_bins))
    

    unique_bins = set(bins)
    

    downsampled_points = []
    for bin in unique_bins:

        idxs = [i for i, b in enumerate(bins) if b == bin]

        downsampled_points.append(points[idxs[0]])

    downsampled_points = np.array(downsampled_points)

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)

    return downsampled_points

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="data_generation",
)
def main(cfg):
    simulator = get_simulator(cfg)
    save_path = os.path.join(cfg.dataset_path, simulator.scene_name)
    planner = get_planner(cfg, device)
    voxel_map = VoxelMap(cfg.mapper, simulator.bbox, device,simulator.valid_mask)

    iter = 0
    converged = 0
    train_views = []
    # set up gui messages
    mp.set_start_method("spawn")
    if cfg.use_gui:
        init_event = mp.Event()
        q_mapper2gui = mp.Queue()
        q_gui2mapper = mp.Queue()
        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
        }
        gui_process = mp.Process(
            target=gui.run,
            args=(
                init_event,
                cfg.gui,
                params_gui,
            ),
        )
        gui_process.start()
        init_event.wait()

    # map free space in the scene
    while iter < cfg.max_iter and converged < cfg.converged_step:
        path = planner.plan([None, voxel_map], simulator, None)
        pose = path[-1]
        dataframe = simulator.simulate(torch.tensor(pose), require_gt=True)
        camera_frame = Camera.init_from_mapper(iter, dataframe)
        train_views.append(dataframe)
        dataframe = {k: v.to(device) for k, v in dataframe.items()}
        voxel_state_old = voxel_map.unexplored_mask.clone()
        voxel_map.update(dataframe)
        voxel_state_new = voxel_map.unexplored_mask.clone()
        changes = track_changes(voxel_state_old, voxel_state_new)
        if changes == 0:
            converged += 1
        else:
            converged = 0
        iter += 1
        if cfg.use_gui:
            q_mapper2gui.put(
                Mapper2Gui(
                    current_frame=camera_frame,
                    gaussians=None,
                    voxels=voxel_map,
                )
            )
            time.sleep(0.5)
    


    test_views = generate_test_views(voxel_map, cfg.num_views)
   

    record_data(save_path, simulator, train_views,test_views)


def record_data(path, simulator, train_views,test_views):
         # Save intrinsic parameters (camera model, width, height, focal length, cx, cy)
    H, W = simulator.resolution
    fx = W * simulator.intrinsic[0,0]
    focal_length = fx # Assuming fx = fy = focal length
    cx = simulator.intrinsic[0,2] *  W # Principal point x
    cy = simulator.intrinsic[1,2] *  H   # Principal point y


    print(f"\n ---------- generating {len(test_views)} test views ----------")
    test_path = os.path.join(path,"test")
    train_path = os.path.join(path,"train")
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)


    os.makedirs(f"{test_path}/images", exist_ok=True)
    os.makedirs(f"{test_path}/depth", exist_ok=True)

    # os.makedirs(f"{train_path}/images", exist_ok=True)
    # os.makedirs(f"{train_path}/depth", exist_ok=True)


    with open(f"{train_path}/cameras.txt", 'w') as f:
        f.write(f"1 PINHOLE {W} {H} {focal_length} {focal_length} {cx} {cy}\n")

    with open(f"{test_path}/images.txt", 'w') as ffile:
        for i, pose in tqdm(enumerate(test_views), total=len(test_views)):
            dataframe = simulator.simulate(torch.tensor(pose))
            rgb = dataframe["rgb"]
            depth = dataframe["depth"]

            rgb_img = Image.fromarray(
                (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode="RGB"
            )
            rgb_img.save(os.path.join(test_path, "images/{:05}.png".format(i)))
            depth_img = Image.fromarray(
                (depth.squeeze(0).numpy() / 10 * 255).astype(np.uint8), mode="L"
            )
            depth_img.save(os.path.join(test_path, "depth/{:05}.png".format(i)))

            # Extract rotation matrix and translation vector
            rotation_matrix = pose[:3, :3]
            translation_vector = pose[:3, 3]

            # Compute the inverse of the pose (rotate Z -> X)
            rotation_inv = rotation_matrix.T  # Inverse of rotation matrix
            translation_inv = -rotation_inv @ translation_vector  # Inverse of translation

            # Convert the inverse rotation matrix to quaternion
            rotation = R.from_matrix(rotation_inv)  # Using inverse rotation matrix
            qx, qy, qz, qw = rotation.as_quat()

         # Write image entry in COLMAP format (with an empty line after each entry)
            ffile.write(f"{i} {qw} {qx} {qy} {qz} {translation_inv[0]} {translation_inv[1]} {translation_inv[2]} 1 {i:05}.png\n")
            ffile.write("0 0 0\n")  # Empty line after each entry as per COLMAP format


 
    # Save extrinsic parameters (rotation and translation)
    pts, face_indices = simulator.mesh.sample(500000, return_index=True)

  
    normals = simulator.mesh.face_normals[face_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts))

 
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    o3d.io.write_point_cloud(train_path + "/points3D.ply", pcd)

    print(f"COLMAP format data saved to {test_path}/cameras.txt and {test_path}/images.txt")


def track_changes(old_state, new_state):
    unknown_old = old_state == 0
    unknown_new = new_state == 0
    change_mask = unknown_old != unknown_new
    changes = torch.sum(change_mask)
    return changes

def get_six_direction_rotations():
    dirs = [
        np.array([ 1,  0,  0]),
        np.array([-1,  0,  0]),
        np.array([ 0,  1,  0]),
        np.array([ 0, -1,  0]),
        np.array([ 0,  0,  1]),
        np.array([ 0,  0, -1])
    ]
    rots = []
    up = np.array([0, 0, 1], dtype=float)

    Rz180 = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    for d in dirs:
        forward = d.astype(float)
        forward /= np.linalg.norm(forward)
        up2 = up if abs(np.dot(forward, up)) <= 0.9 else np.array([0,1,0], dtype=float)
        right = np.cross(up2, forward)
        right /= np.linalg.norm(right)
        new_up = np.cross(forward, right)
        new_up /= np.linalg.norm(new_up)
        R_cam = np.vstack([right, new_up, forward]).T
 
        R_cam =  R_cam @ Rz180
        rots.append(R_cam)
    return rots

def generate_test_views(voxel_map, num_views):

    voxel_center = voxel_map.voxel_centers.cpu().numpy()
    # voxel_states = voxel_map.voxel_states
    voxel_size = voxel_map.size.cpu().numpy()

    free_mask = voxel_map.free_mask.cpu().numpy()
    num_free_voxel = np.sum(free_mask)
    points_needed = max(num_views, num_free_voxel)

    num_per_voxel = np.ceil(points_needed / num_free_voxel).astype(int)
    select_points = downsample_irregular_points(voxel_center[free_mask])
    # voxel_min = voxel_center[free_mask] - 0.5 * voxel_size
    # voxel_max = voxel_min + voxel_size

    # repeated_voxel_min = np.repeat(voxel_min, num_per_voxel, axis=0)
    # repeated_voxel_max = np.repeat(voxel_max, num_per_voxel, axis=0)

    # points = np.random.uniform(
    #     repeated_voxel_min, repeated_voxel_max, size=(len(repeated_voxel_min), 3)
    # )
    # if len(points) > num_views:
    #     indices = np.random.choice(len(points), num_views, replace=False)
    #     sampled_points = points[indices]
    # else:
    #     sampled_points = points

    rots = get_six_direction_rotations()
    views = []
    for pt in select_points:
        for R_cam in rots:
            T = np.eye(4, dtype=float)
            T[:3, 3] = pt
            T[:3, :3] = R_cam
            views.append(T)
    Ts = np.stack(views, axis=0)

    return torch.tensor(Ts).type(torch.float32)


if __name__ == "__main__":
    main()
