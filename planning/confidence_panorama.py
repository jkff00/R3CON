from .plan_base import *
from utils.common import TextColors
from utils.operations import GaussianRenderer
from mapping.renderability import FibonacciSphere

from utils.operations import vec_to_rot,get_visible_points

class ConfidencePano(PlanBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.render_ratio = cfg.render_ratio
        self.explore_weight = cfg.explore_weight

        self.fibonaccisphere = FibonacciSphere(theta_res= 10.0, fov_x= 60.0,fov_y=60.0)#set to your own camera fov

    @torch.no_grad
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator,sampler,hash_map,metric_map):
    
        t_utility = 0
        depth_range = simulator.depth_range

        extrinsics = candidates.to(self.device)
        t_candidate = extrinsics[:,:3,3]
        extrinsics[:, :3, :3] = torch.eye(3, device=self.device).unsqueeze(0)
                # main planning section

        depths = []
        ray_norm = []
        metrics_points = []
        metric_view_points = []
        metric_view_mean = []
        metric_pts  = []
        t_utility_start = time.time()
    
        for i in tqdm(
            range(len(extrinsics)),
            desc=f" {TextColors.CYAN}Evaluate View Candidates{TextColors.RESET}",
        ):
            out = sampler.render_simple(hash_map.points,hash_map.opacities,hash_map.scales,hash_map.rotations,extrinsics[i],hash_map.colors)

            depths.append(out["depth"][0])
            data_dict = get_visible_points(out=out,c2w=extrinsics[i],points=hash_map.points)
            metric = 1-metric_map.query(data_dict)
            ray_norm.append(data_dict["normaliz_ray_points_view"])
            metrics_points.append(metric)
            metric_view_points.append(torch.sum(metric))
            metric_view_mean.append(torch.mean(metric))
            metric_pts.append(hash_map.points[out["contrib"].long()])
        t_time_mean = (time.time()-t_utility_start)/len(extrinsics)

        depth_tensor = torch.stack(depths)
 
        exploit_util = torch.stack(metric_view_points)
        #add
        metric_view_mean_tensor = torch.stack(metric_view_mean)

        depth_tensor[depth_tensor < 0.001] = 10000  # unseen surfaces
        depth_tensor = torch.clamp(
            depth_tensor, min=depth_range[0], max=depth_range[1]
        )

        visible_mask,voxel_dir = voxel_map.cal_visible_mask_pano(t_candidate,depth_tensor)

        explore_util = torch.sum(visible_mask,dim=1)
        
        exploit_util[torch.isnan(exploit_util)] = 0.0
        explore_util[torch.isnan(explore_util)] = 0.0

        utility = self.explore_weight * explore_util + exploit_util
        metric_view_mean_tensor[torch.isnan(metric_view_mean_tensor)] = 0.0
        t_utility += time.time() - t_utility_start
        return {"utility":utility,
                "time": t_utility,
                "metric": metrics_points,
                "visible": visible_mask,
                "ray": ray_norm,
                "ray_voxel": voxel_dir,
                "t_mean": t_time_mean,
                "metric_mean":metric_view_mean_tensor.max(),
                }
    

    @torch.no_grad
    def view_choose(self,metric_tensor, visible_mask, ray_norm,voxel_dir):
        t1 = time.time()

        voxel_norm = voxel_dir[visible_mask]

        voxel_metric = torch.ones(len(voxel_norm),
                        dtype=metric_tensor.dtype,
                        device=metric_tensor.device)*self.explore_weight

        new_metric = torch.cat(
    [metric_tensor, voxel_metric], dim=0)
        new_norm = torch.cat(
    [ray_norm, voxel_norm], dim=0)    
        
        indice,_ = self.fibonaccisphere.query(new_norm)
        mean_value = self.fibonaccisphere.reduce_bins(indice,new_metric)
        dir_c2w = self.fibonaccisphere.view_choose(mean_value)
        print("view_choose:",time.time()-t1)
        

        return vec_to_rot(dir_c2w),self.fibonaccisphere.points,mean_value/torch.max(mean_value)


