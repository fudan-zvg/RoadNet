# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Union

import numpy as np
from pyquaternion import Quaternion
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmengine.dataset import BaseDataset


@DATASETS.register_module()
class CenterlineNuScenesDataset(BaseDataset):
    r"""NuScenes Dataset.
    This datset only add camera intrinsics and extrinsics to the results.
    """
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }
    
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False, 
                 grid_conf=None,
                 bz_grid_conf=None,
                 landmark_thresholds=[1, 3, 5, 8, 10],
                 reach_thresholds=[1, 2, 3, 4, 5],
                 backend_args = None,
                 **kwargs
                 ):
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.modality = modality
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.interval = grid_conf['xbound'][-1]
        self.landmark_thresholds = landmark_thresholds
        self.reach_thresholds = reach_thresholds
        self.pkl = ann_file

    def parse_ann_info(self, info: dict) -> dict:
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.METAINFO['classes']:
                gt_labels_3d.append(self.METAINFO['classes'].index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results
    
    def parse_data_info(self, info: dict) -> dict:
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            # lidar2ego_translation=info['lidar2ego_translation'],
            # lidar2ego_rotation=info['lidar2ego_rotation'],
            # ego2global_translation=info['ego2global_translation'],
            # ego2global_rotation=info['ego2global_rotation']
        )
        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego_t = info['lidar2ego_translation']
        lidar2ego_rt = np.eye(4)
        lidar2ego_rt[:3, :3] = lidar2ego_r
        lidar2ego_rt[:3, 3] = lidar2ego_t

        input_dict.update(
            dict(
                lidar2ego=lidar2ego_rt
            )
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics 
                ))

        # if not self.test_mode:
        #     annos = self.parse_ann_info(info)
        #     input_dict['ann_info'] = annos
        # gt_lines = info['lines']
        # input_dict['gt_lines_coord'] = gt_lines['boxes']
        # input_dict['gt_lines_label'] = gt_lines['labels']
        if 'center_lines' in info.keys():
            input_dict['center_lines'] = info['center_lines']
        # input_dict['raw_center_lines'] = PryCenterLine(info['center_lines'])
        return input_dict