# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
from .vovnetcp_syncbn import VoVNetCPSyncBN
from .resnet import ResNetV1c
__all__ = ['VoVNet', 'VoVNetCP', 'ResNetV1c', 'VoVNetCPSyncBN']

