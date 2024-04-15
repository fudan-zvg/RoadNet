import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import mmcv
import numpy as np
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmcv.cnn import ConvModule
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
# from  import LiftSplatShoot, Up
from ..utils.LiftSplatShoot import LiftSplatShoot, LiftSplatShootEgo, Up, GTDepthLiftSplatShoot
from ..utils.LiftSplatShoot_sync import LiftSplatShootEgoSyncBN
from mmdet3d.models import builder

@DETECTORS.register_module()
class LssSegEgo(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lss_cfg=None,
                 grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 ):
        super(LssSegEgo, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                         pts_middle_encoder, pts_fusion_layer,
                                         img_backbone, pts_backbone, img_neck, pts_neck,
                                         pts_bbox_head, img_roi_head, img_rpn_head,
                                         train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.bins = 200
        self.start = 2001
        self.padding = 2002
        self.end = 2000
        self.category_start = 1500
        self.no_known = 2002  # n/a and padding share the same label to be eliminated from loss calculation
        self.noise = 1998
        self.vis_cfg = vis_cfg

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats

    def forward_pts_train(self,
                          bev_feats,
                          center_seg,
                          img_metas=None,
                          ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outputs = self.pts_bbox_head(bev_feats)
        loss_inputs = [outputs, center_seg.unsqueeze(1)]
        losses = self.pts_bbox_head.losses(*loss_inputs)
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      center_seg=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(bev_feats, center_seg, img_metas)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, bev_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outputs = self.pts_bbox_head(bev_feats)
        seg_pred = outputs.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # self.vis_centerline(seg_pred[0], img_metas[0], 'centerline_d8_val_20')
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        seg_list = [dict(gt_seg_maps=img_meta['sample_idx']) for img_meta in img_metas]
        seg_preds = self.simple_test_pts(bev_feats, img_metas)
        for result_dict, seg_pred in zip(seg_list, seg_preds):
            result_dict['centerline_seg'] = seg_pred
        return seg_list

