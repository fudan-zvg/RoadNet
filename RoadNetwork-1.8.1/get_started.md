# Get Started
## Prerequisities
Our code is tested on the following environment:
- Linux
- Python 3.8
- PyTorch 1.8.1
- Cudatoolkit 11.3
- mmdet3d 0.17.1


PyTorch version **1.8.0** or higher and **mmdetection3d==0.17.1**.

## Installation
Setup Environment
```
conda create -n roadnet python=3.8 -y
conda activate roadnet
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Install [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) correctly. please visit the official [documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).
Install MMDetection
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```
Install MMSegmentation
```
pip install mmsegmentation==0.20.2
pip install einops
pip install bezier==0.11.0
```
Install MMDetection3d
```
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```
Add our projects to mmdetection3d projects
```
cd ${any path outside mmdetection3d}
git clone git@github.com:fudan-zvg/RoadNet.git
cd RoadNet/RoadNetwork-1.8.1/
ln -s {mmdetection3d_path} ./mmdetection3d
```
## Data Preparation
```
mkdir data
ln -s {nuscenes_path} ./data/nuscenes
```
Please refer to nuScenes for initial preparation
Run the following code to generate `.pkl` file.
```
python tools/create_data_centerline.py nuscenes
python tools/create_data_pon_centerline.py nuscenes
```

## Checkpoint Preparation
```
mkdir ckpts
```
Download ResNet-50 Deeplab-V3-Plus [checkpoint](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth) from MMSegmentation.
