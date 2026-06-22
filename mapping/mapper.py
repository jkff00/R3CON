import torch
import time

from utils.operations import *
from utils.common import Camera, Mapper2Gui, FakeQueue, TextColors
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap
#add jxf
from mapping.camera_new_utils import SimpleCamera
from mapping.renderability import PointMetrics,VoxelHashMap
import open3d as o3d
import torchvision
import os
#add jxf

class IncrementalMapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # map instance
        self.gaussian_map = None
        self.voxel_map = None

        # gui related
        self.use_gui = False
        self.q_mapper2gui = FakeQueue()
        self.q_gui2mapper = FakeQueue()
        self.pause = False
        self.init = False

        self.camera = []

    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map
    #add jxf
    def qvec2rotmat(self,qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    def load_test_views(self,path):
        c2w_list = []

        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    elems = fid.readline().split()
                    rot = self.qvec2rotmat(qvec)

                    # construct w2c homogeneous
                    c2w = np.eye(4)
                    c2w[:3, :3] = rot.T
                    c2w[:3, 3] = (-rot.T @ tvec)
                    c2w_list.append(c2w)
        self.camera = c2w_list


    def load_metric_processor(self,cfg):
        #add jxf init for renderability
        self.metric_map = PointMetrics(Fx=self.simulator.intrinsic[0,0])
        self.hash_map = VoxelHashMap (float(cfg.experiment.voxel_size))
        self.observer = SimpleCamera (width=self.simulator.resolution[1], height=self.simulator.resolution[0],fx=self.simulator.intrinsic[0,0],fy=self.simulator.intrinsic[1,1],scale_val=self.hash_map.scale_val,tag="perspective")
        self.sampler = SimpleCamera (width=self.simulator.resolution[1],height=self.simulator.resolution[1],scale_val=self.hash_map.scale_val,tag="spherical")
        
        #add jxf
    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
        self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device,self.simulator.valid_mask)

    def get_new_dataframe(self, i):
        # return way points to the nbv
        if self.planner.use_perspective:
            path = self.planner.plan(self.current_map, self.simulator, self.recorder,[self.observer,self.hash_map,self.metric_map])#
        else:
            path = self.planner.plan(self.current_map, self.simulator, self.recorder,[self.sampler,self.hash_map,self.metric_map])#
        print("path length:", len(path))
        # for visualization only
        if self.use_gui:
            for pose in path:
                #add jxf
                metric = None
                if not self.planner.use_confidence:
                    extrinsic = pose.to(self.hash_map.points.device)
                    out = self.observer.render_simple(self.hash_map.points,self.hash_map.opacities,
                                        self.hash_map.scales,self.hash_map.rotations,
                                        extrinsic,self.hash_map.colors)
                    data_dict = get_visible_points(out=out,c2w=extrinsic,points=self.hash_map.points)
                    metrics_points = self.metric_map.query(data_dict)
                    out = self.observer.render_simple(self.hash_map.points[data_dict["hidden_points"]], self.hash_map.opacities[data_dict["hidden_points"]], 
                                                self.hash_map.scales[data_dict["hidden_points"]], self.hash_map.rotations[data_dict["hidden_points"]], 
                                                extrinsic, jet_colormap(1-metrics_points).float(),render_only= True)
                    metric = out["render"]* 255

                dataframe = self.simulator.simulate(pose)
                camera_frame = Camera.init_from_mapper(None, dataframe,metric=metric)
                self.q_mapper2gui.put(
                    Mapper2Gui(
                        current_frame=camera_frame,
                    )
                )
                time.sleep(0.05)

        # dataframe at nbv as keyframe
        dataframe = self.simulator.simulate(path[-1])
        camera_frame = Camera.init_from_mapper(i, dataframe)
        self.q_mapper2gui.put(
            Mapper2Gui(
                current_frame=camera_frame,
            )
        )
        return dataframe

    def run(self):
        torch.cuda.empty_cache()
        self.init_map()
        frame_id = 0

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )
        while self.recorder is None or self.recorder.is_alive:
            # pause information from gui
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            dataframe = self.get_new_dataframe(frame_id)
            dataframe = {k: v.to(self.device) for k, v in dataframe.items()}

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")
            
            t_mapper_start = time.time()
            # update gaussian map
 
            if self.planner.use_confidence:
                self.gaussian_map.update(dataframe)
            else:

                new_pcd_world = compute_valid_point_cloud(dataframe)
                
                self.hash_map.update(new_pcd_world)
                self.metric_map.add_new_points(self.hash_map.new_num_voxels)
                out = self.observer.render_simple(self.hash_map.points,self.hash_map.opacities,
                                                self.hash_map.scales,self.hash_map.rotations,
                                                dataframe["extrinsic"],self.hash_map.colors)
                data_dict = get_visible_points(out,dataframe["rgb"],dataframe["extrinsic"],self.hash_map.points)
                self.metric_map.update_all(data_dict)
            

            end_time = time.time()-t_mapper_start
            # update voxel map
            self.voxel_map.update(dataframe)

            t_mapper = time.time() - t_mapper_start
            frame_id += 1
            self.q_mapper2gui.put(
                Mapper2Gui(
                    # gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                    # camera=self.camera,
                 
                )
            )
            # self.camera = None

            # update recorder or/and save map
            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper)
                # self.recorder.update_time("mapping", 1.0)
                self.recorder.log()
                self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                self.recorder.save_update_time(end_time)
                self.recorder.save_peak_memory()
                if self.recorder.require_record:
                    # self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")
                    self.recorder.save_map(self.gaussian_map, f"{0:03}")
                    self.recorder.save_path()
            time.sleep(0.1)

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )
