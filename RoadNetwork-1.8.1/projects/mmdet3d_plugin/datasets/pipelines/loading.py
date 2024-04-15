# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np
import os
import torch
import random
import math
import cv2
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet.datasets.pipelines import LoadAnnotations
from mmdet3d.core.bbox import Coord3DMode
from .depth_map_utils import fill_in_multiscale

from .centerline_utils import SceneGraph, sentance2seq, sentance2bzseq, sentance2bzseq2
from projects.mmdet3d_plugin.core.centerline import NusOrederedBzCenterLine, OrderedBzSceneGraph
LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']

@PIPELINES.register_module()
class LoadCenterlineSegFromPkl(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """
    def __init__(self, grid_conf, thickness=2, data_root=None):
        self.line = 255
        self.data_root = data_root
        self.thickness = thickness
        self.grid_conf = grid_conf
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate((self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        centerline_seg = np.zeros((int(self.nx[1]), int(self.nx[0])))
        center_lines = results['center_lines']['centerlines']
        for i in range(len(center_lines)):
            center_line = center_lines[i]
            inbev_x = np.logical_and(center_line[:,0] < self.pc_range[3], center_line[:,0] >= self.pc_range[0])
            inbev_y = np.logical_and(center_line[:,1] < self.pc_range[4], center_line[:,1] >= self.pc_range[1])
            inbev_xy = np.logical_and(inbev_x, inbev_y)
            center_line = (center_line[inbev_xy, :] - self.pc_range[:3]) / self.dx
            center_line = np.floor(center_line).astype(np.int)
            for pt_i in range(len(center_line)-1):
                cv2.line(centerline_seg, tuple(center_line[pt_i, :2]), tuple(center_line[pt_i+1, :2]), self.line, self.thickness)
        if self.data_root:
            filename = os.path.join(self.data_root, results['sample_idx'] + '.png')
            cv2.imwrite(filename, centerline_seg)
        centerline_seg[centerline_seg==self.line] = 1
        results['center_seg'] = centerline_seg.astype(np.int64)
        return results
    
    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]))
        return dx, bx, nx

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.line}, '
        return repr_str


@PIPELINES.register_module()
class LoadNusOrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """
    def __init__(self, grid_conf, bz_grid_conf):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = NusOrederedBzCenterLine(results['center_lines'], self.grid_conf, self.bz_grid_conf)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class TransformOrderedBzLane2Graph(object):
    def __init__(self, n_control=3, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj, centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(orderedDFS=self.order)
        centerline_sequence = sentance2bzseq(scene_sentance_list,centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack([centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack([centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control
        return results
