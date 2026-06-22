import numpy as np
import habitat_sim
import torch


SENSOR_TYPE = {
    "color": habitat_sim.SensorType.COLOR,
    "depth": habitat_sim.SensorType.DEPTH,
    "semantic": habitat_sim.SensorType.SEMANTIC,
}

#add jxf
def precompute_uv(intrinsic,H, W):
    """
    Precompute u, v pixel coordinates based on image dimensions and intrinsic matrix.
    
    Args:
        H (int): Image height.
        W (int): Image width.
        intrinsic (torch.Tensor): Camera intrinsic matrix (3x3).
        
    Returns:
        uv (torch.Tensor): Precomputed (2, H * W) tensor of u, v pixel coordinates.
    """
    # Create a grid of pixel coordinates (i, j) where i is row and j is column
    v, u = torch.meshgrid(
    torch.arange(H, device=intrinsic.device),
    torch.arange(W, device=intrinsic.device),
    indexing='ij'
)
    
    uv_homogeneous = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    uv_homogeneous = uv_homogeneous.reshape(-1, 3).T  # (3, H*W)

    uv_homogeneous = uv_homogeneous.to(dtype=intrinsic.dtype)
    return intrinsic @ uv_homogeneous


def compute_camera_intrinsic(h, w, vfov, hfov, normalize=True):
    vfov_rad = np.radians(vfov)
    hfov_rad = np.radians(hfov)

    fx = (w / 2) / np.tan(hfov_rad / 2)
    fy = (h / 2) / np.tan(vfov_rad / 2)

    cx = w / 2
    cy = h / 2

    if normalize:
        fx /= w
        cx /= w
        fy /= h
        cy /= h

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    return torch.from_numpy(K)


def opencv_to_opengl_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def opengl_to_opencv_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
