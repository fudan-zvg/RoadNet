# Get Started
## Prerequisities
Our code is tested on the following environment:
- Linux
- Python 3.8
- PyTorch 2.0.1
- Cudatoolkit 11.7
- mmdet3d 1.4.0


PyTorch version **2.0** or higher and mmdetection3d, which relies on **mmengine**, are mandatory requirements.

## Installation
Setup Environment
```
conda create -n roadnet python=3.8 -y
conda activate roadnet
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) correctly. please visit the official [documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).
```
git clone git@github.com:open-mmlab/mmdetection3d.git
cd mmdetection 3d
git checkout v1.4.0
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
pip install -v -e .
```
Install some extra envirnment
```
pip install mmsegmentation
pip install einops
pip install bezier # for bezier curve
```
Add our projects to mmdetection3d projects
```
cd ${any path outside mmdetection3d}
git clone git@github.com:fudan-zvg/RoadNet.git
cp -r RoadNet/RoadNetwork-2.0.1/ mmdetection3d/projects/RoadNetwork/
```
## Data Preparation
Please refer to nuScenes for initial preparation
Run the following code to generate `.pkl` file.
```
python projects/RoadNetwork/tools/create_data_pon_centerline.py nuscenes
```

## Checkpoint Preparation
```
mkdir ckpts
```
Download ResNet-50 Deeplab-V3-Plus [checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth) from MMSegmentation.
