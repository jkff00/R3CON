from .plan_base import *
from utils.common import TextColors
from utils.operations import GaussianRenderer
from utils.operations import get_visible_points,vec_to_rot
import torch.nn.functional as F

class ConfidencePers(PlanBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.render_ratio = cfg.render_ratio
        self.explore_weight = cfg.explore_weight

    @torch.no_grad
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator,sampler,hash_map,metric_map):
        t_utility = 0
        render_resolution = np.round(self.render_ratio * simulator.resolution).astype(
            int
        )
        h, w = render_resolution
        depth_range = simulator.depth_range
        extrinsics = candidates.to(self.device)
        intrinsics = repeat(simulator.intrinsic, " h w -> v h w", v=len(candidates)).to(
            self.device
        )
        
        explore_util = torch.zeros(len(candidates))
        exploit_util = torch.zeros(len(candidates))

        require_valid_mask = simulator.has_missing_surface
        metric_view_mean = []
        # main planning section
        for i in tqdm(
            range(len(extrinsics)),
            desc=f" {TextColors.CYAN}Evaluate View Candidates{TextColors.RESET}",
        ):
            t_utility_start = time.time()
            out = sampler.render_simple(hash_map.points,hash_map.opacities,hash_map.scales,hash_map.rotations,extrinsics[i],hash_map.colors)

            data_dict = get_visible_points(out=out,c2w=extrinsics[i],points=hash_map.points)
            metric = 1-metric_map.query(data_dict)


            depths =  F.interpolate(out["depth"][0].unsqueeze(0).unsqueeze(0),
                      size=(128, 128),
                      mode='bilinear',
                      align_corners=False).squeeze()

            # due to missing surfaces issue in dataset,
            # we use simulator to get a valid mask at view candidates
            # to ignore the value at missing surfaces.
            if require_valid_mask:
                t_simulator_start = time.time()
                valid_mask = simulator.simulate(
                    extrinsics[i].cpu(), valid_mask_only=True
                )
                valid_mask = cv2.resize(
                    valid_mask.astype(np.uint8),
                    render_resolution,
                    interpolation=cv2.INTER_NEAREST,
                )
                valid_mask = torch.tensor(valid_mask).bool()
                t_simulator = time.time() - t_simulator_start
            else:
                valid_mask = torch.ones(*render_resolution).bool()
                t_simulator = 0

            # exploration utility
            depth_voxel = depths.clone()
            depth_voxel[depth_voxel < 0.001] = 10000  # unseen surfaces
            depth_voxel = torch.clamp(
                depth_voxel, min=depth_range[0], max=depth_range[1]
            )
            depth_voxel[~valid_mask] = -1.0
            visible_mask = voxel_map.cal_visible_mask(
                extrinsics[i],
                intrinsics[i],
                depth_voxel,
            )

            unexp_mask = voxel_map.unexplored_mask
            visible_unexp_mask = visible_mask & unexp_mask
            explore_util[i] = torch.sum(visible_unexp_mask)

            # exploitation utility
            exploit_util[i] = torch.sum(metric)
            metric_view_mean.append(torch.mean(metric))

            t_utility += time.time() - t_utility_start - t_simulator

        exploit_util[torch.isnan(exploit_util)] = 0.0
        explore_util[torch.isnan(explore_util)] = 0.0
        

        utility = self.explore_weight * explore_util + exploit_util
 
        metric_view_mean_tensor = torch.stack(metric_view_mean)
        metric_view_mean_tensor[torch.isnan(metric_view_mean_tensor)] = 0.0
        return {"utility":utility.cuda(),
                "time": t_utility,
                "metric_mean":metric_view_mean_tensor.max(),}
    @torch.no_grad
    def view_choose(self,):
        return None
