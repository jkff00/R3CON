# R3CON: Radiance-Field-Free Active Reconstruction via Renderability (ECCV 2026)

Xiaofeng Jin, Matteo Frosi, Yiran Guo, Matteo Matteucci

Politecnico di Milano

<a target="_blank" href="https://arxiv.org/abs/2601.07484">
    <img src="https://img.shields.io/badge/arXiv-2412.17769-b31b1b.svg" alt="arXiv Paper">
</a>



<!-- <a target="_blank" href="https://arxiv.org/abs/2412.17769">
    <img src="https://img.shields.io/badge/arXiv-2412.17769-b31b1b.svg" alt="arXiv Paper">
</a> -->

▶️ **Demo video** (MP4): 
[![Demo Video](https://github.com/jkff00/r3recon/releases/download/v1/2026-01-01.01-04-29.png)](https://github.com/jkff00/r3recon/releases/download/v1/paper-demo.mp4)


## Setup
We test the following setup on Ubuntu20 with CUDA11.8. 

Clone the repo:
```
git clone https://github.com/jkff00/R3CON.git
cd R3CON
```

(optional) For different CUDA versions in your machine, you might need to change the corresponding pytorch version and source in envs/build.sh:
```
# for example for CUDA 12.1, change the source. 
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# you can find more compatible version on https://pytorch.org/get-started/previous-versions/
```
Create and activate environment:
```
bash envs/build.sh
conda activate R3CON
```

## Data

Download full Replica dataset:
```
bash data/replica_download.sh
```
We release the collected data for **all methods under the same view budget**, together with the corresponding **Replica-Dense test views** used in our evaluation.

**Download (OneDrive):** [view_budget_replica_dense.zip](https://polimi365-my.sharepoint.com/:u:/g/personal/11092598_polimi_it/IQDTcHulM-sASLmxWZUKLBnPAYB4DW9WxnXLFrFfcTfI92E?e=RDLvNP)

## Run
Run online mission:
```
python main.py planner=PLANNER_TYPE scene=SCENE_NAME
# example: 
# python main.py planner=confidence_pano scene=replica/office0 use_gui=true
# for perspective version:
# python main.py planner=confidence_perspective scene=replica/office0 use_gui=true
# for the baseline active-gs:
# python main.py planner=confidence scene=replica/office0 use_gui=true

#The result will save as follow:
# R3CON/dataset/DATASET/SCENE_NAME/train/PLANNER_TYPE/ {images/...png images.txt time.txt ...}
#example:
# R3CON/dataset/replica/office0/train/confidence_pano/...
```
If use_gui is set to true, you should be able to see a GUI running.


To visualize the built GS map:
```
python visualize.py -G PATH_TO_GS_MAP
```
## How to use GUI? 
1. Resume/Pause: click to stop or continue online mission.
2. Stop/Record: click to enter camera path recording mode. Any movement of the camera will be recorded and saved in outputs_gui/saved_paths. You can set the ID of camera path to be recorded by choosing number from "Camera Path" and click reset to delete. Click "Fly" in "Camera Follow Options" to control the camera via keyboard (WASD and direction keys). 
3. Camera Pose: You can save individual camera pose by selecting ID of the camera pose and then clicking "Save". Similarly, click "Load" to move the camera to saved camera poses.
4. History Views: move the camera to planned history viewpoints.
5. 3D Objects: click to visualize 3D objects. You can see different submaps in "Voxel Map". "Mesh" is only available if corresponding mesh is also loaded.
6. Rendering Options: click to show rendering results from Gaussian Splatting map. Only one rendering type among "Depth", "Confidence", "Opacity", "Normal" and "D2N" can be visualized at the same time.
## Evaluation
For rendering evaluation, you need to first generate test views for each scene:
```
python data_generation.py scene=SCENE_NAME
# example:
# python data_generation.py scene=replica/office0

#The result will save as follow:
# R3CON/dataset/DATASET/SCENE_NAME/test / {images/...png images.txt}
# R3CON/dataset/DATASET/SCENE_NAME/train/ {cameras.txt, points3D.ply}
```

We also provide a shell script to run all sequences:
```
python run.py --dataset replica
```

Then you can use the standard 3D GS pipeline for reconstruction.
## Citation

```bibtex
@article{jin2026r3,
  title={R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability},
  author={Jin, Xiaofeng and Frosi, Matteo and Guo, Yiran and Matteucci, Matteo},
  journal={arXiv preprint arXiv:2601.07484},
  year={2026}
}
```

## Acknowledgement
Parts of the code are based on [Active-GS](https://github.com/dmar-bonn/active-gs) [OD-GS](https://github.com/esw0116/ODGS). We thank the authors for open-sourcing their code.

## Maintainer
Xiaofeng Jin, xiaofeng.jin@polimi.it
