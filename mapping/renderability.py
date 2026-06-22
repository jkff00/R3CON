import os
import torch
from random import randint
import open3d as o3d
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FibonacciSphere:

    def __init__(self, N=None, theta_res=None, fov_x=None,fov_y=None,device="cuda"):

        if N is None and theta_res is None:
            raise ValueError("must get N or theta_res")

        if N is None:
            # A_point ≈ 2π(1−cos(θ/2))
            # N ≈ 4π / A_point
            N = int(round(2 / (1 - math.cos(theta_res*math.pi / 360))))
            N = max(1, N)


        print("FibonacciSphere: ",N)

        self.N = N
        self.device = device


        self.points = self._generate_fibonacci_points(N).to(device=device)
        self.u,self.v = self.build_plane_basis_from_q_auto(self.points)
        if fov_x is not None:
            self.fov_mask = self.build_fov_matrix(fov_x,fov_y)
            

        #For view_choosen
        self.sum_accum = torch.zeros(N, device=device)
        self.cnt_accum = torch.zeros(N, device=device)


    def _generate_fibonacci_points(self,N):

        i = torch.arange(N, dtype=torch.float32)
        phi = (1 + math.sqrt(5)) / 2  
        z = 1 - 2*(i + 0.5)/N
        theta = torch.acos(z)
        phi_angle = 2 * math.pi * i / phi
        x = torch.cos(phi_angle) * torch.sin(theta)
        y = torch.sin(phi_angle) * torch.sin(theta)
        pts = torch.stack((x, y, z), dim=1)
        pts = pts / pts.norm(dim=1, keepdim=True)
        return pts.to(torch.float32)
    

    def build_fov_matrix(self, fov_x, fov_y):
        fov_x = math.radians(fov_x/2)
        fov_y = math.radians(fov_y/2)
        fov_rad = min(fov_x, fov_y) 

        cosine_sim = torch.matmul(self.points, self.points.T)

        cos_th = math.cos(fov_rad)
        mask = cosine_sim >= cos_th  # (N, N) bool
        return mask
    

    def build_plane_basis_from_q_auto(self,q, eps=1e-8):
        dtype, device = q.dtype, q.device

        ids = torch.abs(q).argmin(dim=1)

        axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype, device=device)
        ref = axes[ids]  # (B,3)

        qdotref = (q * ref).sum(dim=1, keepdim=True)
        a_prime = ref - q * qdotref

        a_norm = a_prime.norm(dim=1, keepdim=True).clamp_min(eps)
        u = a_prime / a_norm
        v = torch.stack([
            q[:, 1] * u[:, 2] - q[:, 2] * u[:, 1],
            q[:, 2] * u[:, 0] - q[:, 0] * u[:, 2],
            q[:, 0] * u[:, 1] - q[:, 1] * u[:, 0]
        ], dim=1)

        return u, v

    def query(self, dirs):
        # (M, N)
        cos_sim = torch.matmul(dirs, self.points.T)
        cos_val, idx = torch.max(cos_sim, dim=1)
        return idx, cos_val
    
    def view_choose(self,values):
        # values: (N,)
        # mask:   (N, N)
        mask = self.fov_mask.float()

        weighted = mask * values      # (N, N)
        sum_vals = weighted.sum(dim=1)   # (N,)
        count_vals = mask.sum(dim=1).clamp(min=1)  # (N,)

        # target_dir = (sum_vals / count_vals).argmax()
        target_dir = (sum_vals).argmax()
        return self.points[target_dir]
    
    def reduce_bins(self, idx, vals):

        self.sum_accum.scatter_add_(0, idx, vals)

        ones = torch.ones_like(vals)
        self.cnt_accum.scatter_add_(0, idx, ones)

        mean_vals = self.sum_accum.clone()
        self.sum_accum.zero_()
        self.cnt_accum.zero_()

        return mean_vals





class PointMetrics:
    def __init__(self, num_points=0, color_dim=3, observation_limitation = 64, Fx = 0,device="cuda"):

        self.N = num_points
        self.C = color_dim
        self.device = device


        self.physical_scale = Fx*0.01/5#the depth of projecting a local regio in five pixels
        self.fibonacci_sphere = FibonacciSphere(N=observation_limitation)
        self.capacity = int(self.fibonacci_sphere.N/2)

  
        self.counts = torch.zeros(num_points, device=device)
        self.color_mean = torch.zeros(num_points, color_dim, device=device)
        self.color_M2 = torch.zeros(num_points, color_dim, color_dim, device=device)


        self.keys_exist = torch.full(
            (num_points,  self.capacity), -1, dtype=torch.int32, device=device
        )

        self.write_ptr = torch.zeros(num_points, dtype=torch.long, device=device)

        self.min_dists = torch.full((num_points,), float("inf"), device=device)
    
    def add_new_points(self, new_num_points: int):

        if new_num_points > 0:
            self.counts = torch.cat([self.counts, torch.zeros(new_num_points, device=self.device)])
            self.color_mean = torch.cat([self.color_mean, torch.zeros(new_num_points, self.C, device=self.device)])
            self.color_M2 = torch.cat([self.color_M2, torch.zeros(new_num_points, self.C, self.C, device=self.device)])


            self.keys_exist = torch.cat([self.keys_exist, torch.full((new_num_points, self.capacity), -1,dtype=torch.long, device=self.device)], dim=0)
            self.write_ptr = torch.cat([self.write_ptr, torch.zeros(new_num_points, dtype=torch.long, device=self.device)], dim=0)


            self.min_dists = torch.cat([self.min_dists, torch.full((new_num_points,), float("inf"), device=self.device)], dim=0)


    def update_color(self, idx_batch: torch.Tensor, color_batch: torch.Tensor):
        """
        idx_batch: (B,) long tensor
        color_batch: (B,C) float tensor
        """

        idx = idx_batch
        cnt = self.counts[idx] + 1  # (B,)
        delta = color_batch - self.color_mean[idx]  # (B,C)

        new_mean = self.color_mean[idx] + delta / cnt.unsqueeze(-1)  # (B,C)

        self.color_M2[idx] += torch.einsum("bi,bj->bij", color_batch - new_mean, delta)

  
        self.counts[idx] = cnt
        self.color_mean[idx] = new_mean

    def update_angle(self, idx_batch: torch.Tensor,new_rays: torch.Tensor):
        """
        keys_exist: (N,M，) 
        new_rays:   (N,3) 
        R: 
          keys_new: (M+K,) 
          keys_added: (K,) 
        """
        keys_new,_ = self.fibonacci_sphere.query(new_rays)

        eq = (keys_new.unsqueeze(1) == self.keys_exist[idx_batch])  # (N,K)
        exists = eq.any(dim=1)  # (N,)
        mask = ~exists

        rows = idx_batch[mask]  # (M,)
        cols = self.write_ptr[idx_batch][mask]  # (M,）

        # clamp [0,32]，32 dummy slot

        self.keys_exist.index_put_((rows, cols), keys_new[mask])
        self.write_ptr.index_put_((rows,), torch.clamp(cols+1, max=self.capacity - 1))


    def update_distance(self, idx_batch: torch.Tensor, dist_batch: torch.Tensor):
        """
        idx_batch: (B,) long tensor
        dist_batch: (B,) float tensor
        """
        self.min_dists[idx_batch] = torch.minimum(self.min_dists[idx_batch], dist_batch)

    def update_all(self, new_data_dict):
        self.update_color(new_data_dict["hidden_points"], new_data_dict["color_gt"])

        self.update_angle(new_data_dict["hidden_points"], new_data_dict["normaliz_ray_points"])

        self.update_distance(new_data_dict["hidden_points"], new_data_dict["dist_points"])

    def compute_final_score(self,color_dists: torch.Tensor,
                            angle_mins: torch.Tensor,
                            dist_scores: torch.Tensor,
                            counts: torch.Tensor):

        h_geo = color_dists.clamp(0.0, 1.0)

        h_res = (1-dist_scores)**(1-angle_mins*h_geo)

        final_scores = h_geo ** (1-angle_mins) * angle_mins * h_res

        final_scores = torch.where(counts >= 1, final_scores, torch.zeros_like(final_scores))

        return final_scores

    def query(self, data_dict):
        """
        idx_batch: (B,) long tensor
        return:
          color_dists: (B,) tensor
            angle_mins: (B,) tensor
          min_dists: (B,) tensor
        """
        idx_batch = data_dict["hidden_points"]
        new_rays = data_dict["normaliz_ray_points"]
        dist_batch = data_dict["dist_points"]

        denom = (self.counts[idx_batch] - 1).clamp_min(1).view(-1, 1, 1)  # (B,1,1)

        covs = self.color_M2[idx_batch] / denom  # (B,C,C)
        trs = covs.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)  # (B,)

        color_dists = 1 - 1.154 * torch.sqrt(trs)  # (B,)


        min_dists = self.min_dists[idx_batch]  # (B,)
        dist_scores = (1 - dist_batch / (min_dists + 1e-8)).clamp_min(0.0)
        dist_bound = min_dists<self.physical_scale
        dist_scores[dist_bound] = 0.0
 
        exist_keys = self.keys_exist[idx_batch] # (B,M)
        valid_mask = exist_keys >= 0

        ray_hist = self.fibonacci_sphere.points[exist_keys]
        cos_theta = torch.bmm(ray_hist, new_rays.unsqueeze(-1)).squeeze(-1)
        cos_theta = torch.where(valid_mask, cos_theta, torch.full_like(cos_theta, -1.0))
        max_vals, max_indices = cos_theta.max(dim=1)
        min_cos_theta = max_vals.clamp_min(0.0)
        batch_idx = torch.arange(idx_batch.shape[0], device=exist_keys.device)
        
        best_keys = exist_keys[batch_idx, max_indices]#get key
        u = self.fibonacci_sphere.u[best_keys]#fibonacci_sphere: [64,3]
        v = self.fibonacci_sphere.v[best_keys]


        ucoords = torch.bmm(ray_hist, u.unsqueeze(-1)).squeeze(-1)  #  u·ray_hist
        vcoords = torch.bmm(ray_hist, v.unsqueeze(-1)).squeeze(-1)
        u_min = torch.where(valid_mask, ucoords, torch.full_like(ucoords, float('inf'))).min(dim=1).values
        u_max = torch.where(valid_mask, ucoords, torch.full_like(ucoords, float('-inf'))).max(dim=1).values
        v_min = torch.where(valid_mask, vcoords, torch.full_like(vcoords, float('inf'))).min(dim=1).values
        v_max = torch.where(valid_mask, vcoords, torch.full_like(vcoords, float('-inf'))).max(dim=1).values
        inside_mask = (u_min <= 0) & (0 <= u_max) & (v_min <= 0) & (0 <= v_max)#B,

        min_cos_theta = torch.where(inside_mask, min_cos_theta, min_cos_theta.square())
   
        color_dists = torch.where(inside_mask, color_dists, color_dists.square())

        return self.compute_final_score(color_dists, min_cos_theta, dist_scores,self.counts[idx_batch])
    

class VoxelHashMap:
    """Voxel-based hash map that keeps Gaussian buffers deduplicated.

    Each voxel can contain at most one Gaussian primitive.  The hash map keeps
    track of tensor buffers storing point attributes and extends them whenever
    a new voxel observation is inserted.
    """

    def __init__(
        self,
        voxel_size = 0.05,
        device = "cuda",
    ) -> None:
        self.voxel_size = voxel_size
        self.scale_val = 0.866*voxel_size
        self.device = device
        self.exsit_keys = torch.empty((0, 1), dtype=torch.long, device=device)
        self.points = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.colors = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.scales = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.opacities = torch.empty((0, 1), dtype=torch.float32, device=device)
        self.rotations = torch.empty((0, 4), dtype=torch.float32, device=device)

        self.default_colors = torch.tensor([0,0,0.0], dtype=torch.float32, device=device)
        self.default_scale = torch.tensor([self.scale_val,self.scale_val,self.scale_val], dtype=torch.float32, device=device)
        self.default_rot = torch.tensor([1.0,0.0,0.0,0.0], dtype=torch.float32, device=device)
        self.default_opacity = torch.tensor([1.0], dtype=torch.float32, device=device)

        self.new_num_voxels = 0
        self.R = 2000
        self.offset = self.R // 2

    def __len__(self) -> int:
        return self.points.shape[0]

    def voxelize_points(self,points: torch.Tensor) -> torch.Tensor:
        """

        Args:
            points (torch.Tensor): (N, 3)
            voxel_size (float): 

        Returns:
            torch.Tensor: (N, 3)
        """
        voxel_coords = torch.floor(points / self.voxel_size).long()
        return voxel_coords

    def expand_points(self,):

        self.colors = torch.cat([self.colors, self.default_colors.repeat(self.new_num_voxels, 1)], dim=0)
        self.scales = torch.cat([self.scales, self.default_scale.repeat(self.new_num_voxels, 1)], dim=0)
        self.rotations = torch.cat([self.rotations, self.default_rot.repeat(self.new_num_voxels, 1)], dim=0)
        self.opacities = torch.cat([self.opacities, self.default_opacity.repeat(self.new_num_voxels, 1)], dim=0)

    def voxel_coords_to_position(self, key: torch.Tensor) -> torch.Tensor:

        i = torch.div(key, self.R * self.R, rounding_mode='floor')
        j = torch.div(key, self.R, rounding_mode='floor') % self.R
        k = key % self.R

        voxel_coords = torch.stack([i, j, k], dim=-1)

        voxel_coords = voxel_coords - self.offset 
        voxel_positions = voxel_coords * self.voxel_size + self.voxel_size / 2.0
        return voxel_positions

    def voxel_coords_to_key(self, voxel_coords: torch.Tensor) -> torch.Tensor:
  
        x, y, z = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
        x = x + self.offset
        y = y + self.offset
        z = z + self.offset

        keys = x * (self.R ** 2) + y * self.R + z
        return keys

    def update(self,points: torch.Tensor) -> torch.Tensor:

        voxel_coords = self.voxelize_points(points)
        
        keys = self.voxel_coords_to_key(voxel_coords)
        unique_keys, unique_idx = torch.unique(keys, return_inverse=True)

        new_voxel_mask = ~torch.isin(unique_keys, self.exsit_keys)  
        new_keys = unique_keys[new_voxel_mask] #N,
        self.new_num_voxels = new_keys.shape[0]

        if self.new_num_voxels>0:
            new_voxel_positions = self.voxel_coords_to_position(new_keys)
            self.points = torch.cat([self.points, new_voxel_positions], dim=0)
            self.exsit_keys = torch.cat([self.exsit_keys, new_keys.unsqueeze(1)], dim=0)
            self.expand_points()





