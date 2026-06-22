from .random import Random
from .exploration import Exploration
from .confidence import Confidence
from .confidence_panorama import ConfidencePano
from .confidence_perspective import ConfidencePers

def get_planner(cfg, device):
    planner_cfg = cfg.planner
    print(planner_cfg.type,"=================")
    if planner_cfg.type == "random":
        return Random(planner_cfg, device)
    elif planner_cfg.type == "exploration":
        return Exploration(planner_cfg, device)
    elif planner_cfg.type == "confidence":
        return Confidence(planner_cfg, device)
    elif planner_cfg.type == "confidence_pano":
        return ConfidencePano(planner_cfg, device)
    elif planner_cfg.type == "confidence_perspective":
        return ConfidencePers(planner_cfg, device)
    else:
        raise NotImplementedError
