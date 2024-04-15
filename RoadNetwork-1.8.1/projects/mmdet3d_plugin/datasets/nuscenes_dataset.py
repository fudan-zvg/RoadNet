# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from collections import OrderedDict
from prettytable import PrettyTable
from mmcv.utils import print_log
from pyquaternion import Quaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from projects.mmdet3d_plugin.core.evaluation import eval_metrics
import os

def seq2nodelist(detection):
    seq = detection['line_seqs'].numpy()
    seq = np.array(seq).reshape(-1, 4)
    node_list = []
    # type_idx_map = {'start': 0, 'continue': 1, 'fork': 2, 'merge': 3}
    idx_type_map = {0: 'start', 1: 'continue', 2: "fork", 3: 'merge'}
    idx = 0
    epsilon = 2

    for i in range(len(seq)):
        node = {'sque_index': None,
                'sque_type': None,
                'fork_from': None,
                'merge_with': None,
                'coord': None}
        label = seq[i][2]
        if label > 3 or label < 0:
            label = 1

        node['coord'] = [seq[i][0], seq[i][1]]
        if label == 3:  # merge
            node['sque_type'] = idx_type_map[3]
            node['sque_index'] = idx
            node['merge_with'] = seq[i][3]

        elif label == 2:  # fork
            node['sque_type'] = idx_type_map[2]
            node['fork_from'] = seq[i][3]

            last_coord = np.array([seq[i - 1][0], seq[i - 1][1]])
            coord = np.array([seq[i][0], seq[i][1]])
            tmp = sum((coord - last_coord) ** 2)
            if tmp < epsilon:  # split fork
                node['sque_index'] = idx
            else:
                idx = idx + 1
                node['sque_index'] = idx

        else:
            node['sque_type'] = idx_type_map[label]
            idx = idx + 1
            node['sque_index'] = idx

        node_list.append(node)

    return node_list

@DATASETS.register_module()
class CenterlineNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.
    This datset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False, 
                 grid_conf=None,
                 bz_grid_conf=None,
                 landmark_thresholds=[1, 3, 5, 8, 10],
                 reach_thresholds=[1, 2, 3, 4, 5],
                 ):
        super().__init__(ann_file=ann_file,
                         pipeline=pipeline,
                         data_root=data_root,
                         classes=classes,
                         load_interval=load_interval,
                         with_velocity=with_velocity,
                         modality=modality,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         test_mode=test_mode,
                         eval_version=eval_version,
                         use_valid_flag=use_valid_flag)
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.interval = grid_conf['xbound'][-1]
        self.landmark_thresholds = landmark_thresholds
        self.reach_thresholds = reach_thresholds
        self.pkl = ann_file

    def get_ann_info(self, index):
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
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
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

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def get_data_info(self, index):
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
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            location=info['location'],
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
        ego_trans = info['ego2global_translation']
        ego_rot = info['ego2global_rotation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego_rot).rotation_matrix
        ego2global[:3, 3] = ego_trans
        input_dict.update(
            dict(
                ego2global=ego2global
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

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        # gt_lines = info['lines']
        # input_dict['gt_lines_coord'] = gt_lines['boxes']
        # input_dict['gt_lines_label'] = gt_lines['labels']
        if 'center_lines' in info.keys():
            input_dict['center_lines'] = info['center_lines']
        # input_dict['raw_center_lines'] = PryCenterLine(info['center_lines'])
        return input_dict

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            node_list = det['pred_node_lists']
            sample_token = self.data_infos[sample_id]['token']
            nusc_annos[sample_token] = node_list
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        # line_results

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0] or 'line_results' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
    
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='ar_reach',
                         result_name='line_results'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        if isinstance(metric, list):
            metric = metric[0]
        datatype, metric = metric.split('_')
        if datatype == 'sar' or datatype == 'nar':
            from projects.mmdet3d_plugin.datasets import BzPlRoadnetReachDistEval as RoadnetReachDistEval
        elif datatype == 'ar':
            from projects.mmdet3d_plugin.datasets import BzRoadnetReachDistEval as RoadnetReachDistEval
        elif datatype == 'clar':
            from projects.mmdet3d_plugin.datasets import ClearBzRoadnetReachDistEval as RoadnetReachDistEval
        else:
            assert False, "Not implement error"
        
        metrics = RoadnetReachDistEval(result_path=result_path, 
                                           data_root=self.data_root,
                                           grid_conf=self.grid_conf,
                                           bz_grid_conf=self.bz_grid_conf,
                                           landmark_thresholds=self.landmark_thresholds,
                                           reach_thresholds=self.reach_thresholds, 
                                           logger=logger, 
                                           pkl=self.pkl
                                           )

        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        landmark_precision = metrics['landmark_precision']
        landmark_recall = metrics['landmark_recall']
        landmark_fscore = metrics['landmark_f_score']
        reach_precision = metrics['reach_precision']
        reach_recall = metrics['reach_recall']
        reach_fscore = metrics['reach_f_score']
        for i, thres in enumerate(self.landmark_thresholds):
            thres_str = float('{:.1f}'.format(thres * self.interval))
            detail['{}/LP_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_precision[i]))
            detail['{}/LR_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_recall[i]))
            detail['{}/LF_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_fscore[i]))
        detail['{}/mLP'.format(metric_prefix)] = float('{:.4f}'.format(landmark_precision.mean()))
        detail['{}/mLR'.format(metric_prefix)] = float('{:.4f}'.format(landmark_recall.mean()))
        detail['{}/mLF'.format(metric_prefix)] = float('{:.4f}'.format(landmark_fscore.mean()))
        for i, thres in enumerate(self.reach_thresholds):
            thres_str = float('{:.1f}'.format(thres * self.interval))
            detail['{}/RP_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_precision[i]))
            detail['{}/RR_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_recall[i]))
            detail['{}/RF_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_fscore[i]))
        detail['{}/mRP'.format(metric_prefix)] = float('{:.4f}'.format(reach_precision.mean()))
        detail['{}/mRR'.format(metric_prefix)] = float('{:.4f}'.format(reach_recall.mean()))
        detail['{}/mRF'.format(metric_prefix)] = float('{:.4f}'.format(reach_fscore.mean()))
        return detail

    def evaluate(self,
                 results,
                 metric='reach',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['line_results'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

@DATASETS.register_module()
class CustomSegNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.
    This datset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self, **kwargs):
        super(CustomSegNuScenesDataset, self).__init__(**kwargs)
        self.SEGCLASSES = ('others', 'centerline')

    def get_ann_info(self, index):
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
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
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
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
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
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
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

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            
        # gt_lines = info['lines']
        # input_dict['gt_lines_coord'] = gt_lines['boxes']
        # input_dict['gt_lines_label'] = gt_lines['labels']
        input_dict['center_lines'] = info['center_lines']
        return input_dict
    
    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['centerline_seg'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 data_root=None,
                 ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        pred_results = [res['centerline_seg'] for res in results]
        gt_seg_maps = [os.path.join(data_root, res['gt_seg_maps']+'.png') for res in results]
        num_classes = len(self.SEGCLASSES)
        ret_metrics = eval_metrics(
                pred_results,
                gt_seg_maps,
                num_classes,
                -1,
                metric,
                label_map={255:1}
                )
        if self.SEGCLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.SEGCLASSES
        
        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })
        return eval_results
