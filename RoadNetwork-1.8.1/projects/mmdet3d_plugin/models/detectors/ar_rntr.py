import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (CameraInstance3DBoxes, LiDARInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmcv.cnn import ConvModule
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
# from  import LiftSplatShoot, Up
from ..utils.LiftSplatShoot import LiftSplatShoot, LiftSplatShootEgo, Up, GTDepthLiftSplatShoot
from ..utils.LiftSplatShoot_sync import LiftSplatShootEgoSyncBN
from projects.mmdet3d_plugin.core import seq2bznodelist, seq2plbznodelist
from mmdet3d.models import builder

@DETECTORS.register_module()
class AR_RNTR(MVXTwoStageDetector):
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
                 freeze_pretrain=True,
                 bev_scale=1.0,
                 lidar_pretrain=False,
                 epsilon=2,
                 max_box_num=100,
                 ):
        super(AR_RNTR, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                         pts_middle_encoder, pts_fusion_layer,
                                         img_backbone, pts_backbone, img_neck, pts_neck,
                                         pts_bbox_head, img_roi_head, img_rpn_head,
                                         train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        # data_aug_conf = {
        #     'final_dim': (128, 352),
        #     'H': 900, 'W': 1600,
        # }
        # self.up = Up(512, 256, scale_factor=2)
        # view_transformers = []
        # view_transformers.append(
        #     LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True))
        # self.view_transformers = nn.ModuleList(view_transformers)
        # self.view_transformers = LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True)
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']
        self.bev_scale = bev_scale

        self.num_center_classes = 576
        self.box_range = 200
        self.coeff_range = 200
        self.num_classes=4
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.start = 574
        self.end = 573
        self.noise_connect = 572
        self.noise_label = 571
        self.noise_coeff = 570
        self.vis_cfg = vis_cfg
        self.epsilon = epsilon
        self.max_box_num = max_box_num

        if lidar_pretrain:
            head_state_dict = torch.load(self.pts_bbox_head.init_cfg.checkpoint)
            if head_state_dict['embedding.position_embeddings.weight'].shape[0] != max_box_num * 6 + 1:
                head_state_dict['embedding.position_embeddings.weight'] = head_state_dict['embedding.position_embeddings.weight'][:max_box_num * 6 + 1, :]
            self.pts_bbox_head.load_state_dict(head_state_dict)
        if freeze_pretrain:
            self.freeze_pretrain()
    
    def freeze_pretrain(self):
        for m in self.img_backbone.parameters():
            m.requires_grad=False
        for m in self.img_neck.parameters():
            m.requires_grad=False
        for m in self.view_transformers.parameters():
            m.requires_grad=False

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
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_lines_coords,
                          gt_lines_labels,
                          gt_lines_connects,
                          gt_lines_coeffs,
                          img_metas,
                          num_coeff, 
                          gt_bboxes_ignore=None):
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
        device = bev_feats[0].device
        box_labels = []
        input_seqs = []
        output_seqs = []
        max_box = max([len(target) for target in gt_lines_coords])
        num_box = max(max_box + 2, self.max_box_num)  # 100
        coeff_dim = num_coeff * 2
        for bi in range(len(gt_lines_coords)):
            box = torch.tensor(gt_lines_coords[bi], device=device).long()
            box = box.reshape(-1,2)
            label = torch.tensor(gt_lines_labels[bi], device=device).long() + self.category_start  # [8,1]
            label = label.reshape(-1,1)
            connect = torch.tensor(gt_lines_connects[bi], device=device).long() + self.connect_start  # [8,1]
            connect = connect.reshape(-1,1)
            coeff = torch.tensor(gt_lines_coeffs[bi], device=device).long() + self.coeff_start  # [8,1]
            coeff = coeff.reshape(-1, coeff_dim)
            box_label = torch.cat([box, label, connect, coeff], dim=-1)  # [8, 5]

            random_box = torch.rand(num_box - box_label.shape[0], 2).to(device)
            random_box = (random_box * (self.box_range - 1)).int()
            random_label = torch.randint(0, self.num_classes, (num_box - box_label.shape[0], 1)).to(label)
            random_label = random_label + self.category_start
            random_connect = torch.randint(0, num_box, (num_box - box_label.shape[0], 1)).to(label)
            random_connect = random_connect + self.connect_start
            random_coeff = torch.rand(num_box - box_label.shape[0], coeff_dim).to(device)
            random_coeff = (random_coeff * (self.coeff_range - 1)).int()
            random_coeff = random_coeff + self.coeff_start
            random_box_label = torch.cat([random_box, random_label.int(), random_connect.int(), random_coeff.int()], dim=-1)  # [92, 5]

            input_seq = torch.cat([box_label, random_box_label], dim=0)  # [100, 5]

            input_seq = torch.cat([torch.ones(1).to(box_label) * self.start, input_seq.flatten()])  # [501]
            input_seqs.append(input_seq.unsqueeze(0))

            output_na = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known  # [92,3]
            output_noise = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known  # [92,1]
            output_noise_connect = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known  # [92,1]
            output_noise_coeff = torch.ones(num_box - box_label.shape[0], coeff_dim).to(input_seq) * self.no_known  # [92,1]
            output_end = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.end  # [92, 1]
            output_seq = torch.cat([output_na, output_noise, output_noise_connect, output_noise_coeff, output_end], dim=-1)  # [92,5]

            output_seq = torch.cat([box_label.flatten(), torch.ones(1).to(box_label) * self.end, output_seq.flatten()])
            output_seqs.append(output_seq.unsqueeze(0))
        input_seqs = torch.cat(input_seqs, dim=0)  # [8,501]
        output_seqs = torch.cat(output_seqs, dim=0)  # [8,501]
        box_labels = output_seqs[:, :-1].flatten()  # [4008]
        outputs = self.pts_bbox_head(bev_feats, input_seqs, img_metas)[-1, :, :-1, :]

        # outputs = outputs[-1]
        # for bi in range(pts_feats[0].shape[0]):
        #     line_seqs = outputs[bi].argmax(dim=-1)
        #     if self.end in line_seqs:
        #         stop_idx = (line_seqs == self.end).nonzero(as_tuple=True)[0][0]
        #     else:
        #         stop_idx = len(line_seqs)
        #     stop_idx = stop_idx // 3 * 3
        #     line_seqs = line_seqs[:stop_idx]
        #     line_seqs = line_seqs.reshape(-1, 3)
        #     pred_points = line_seqs[:, :2].clip(0, 200).float()  # [100,2]
        #     pred_labels = line_seqs[:, 2].unsqueeze(-1) - 1500
        #     self.vis_linepts(pred_points.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), img_metas[bi], 'train_200', 'pred')
        #     self.vis_linepts(gt_lines_coords[bi], gt_lines_labels[bi], img_metas[bi], 'train_200', 'gt')
        clause_length = 4 + coeff_dim
        outputs = outputs.reshape(-1, self.num_center_classes)  # [602, 2003] last layer
        outputs_pos = torch.cat([outputs[::clause_length, :], outputs[1::clause_length, :]])
        outputs_cls = outputs[2::clause_length, :]
        outputs_connects = outputs[3::clause_length, :]
        outputs_coeffs = torch.cat([outputs[k::clause_length, :] for k in range(4, clause_length)])
        inputs_pos = torch.cat([box_labels[::clause_length], box_labels[1::clause_length]])
        inputs_cls = box_labels[2::clause_length]
        inputs_connects = box_labels[3::clause_length]
        inputs_coeffs = torch.cat([box_labels[k::clause_length] for k in range(4, clause_length)])

        gt_coords = [inputs_pos[inputs_pos != self.no_known].long()]
        gt_labels = [inputs_cls[inputs_cls != self.no_known].long()]
        gt_connects = [inputs_connects[inputs_connects != self.no_known].long()]
        gt_coeffs = [inputs_coeffs[inputs_coeffs != self.no_known].long()]

        preds_dicts = dict(
            preds_coords=[outputs_pos[inputs_pos != self.no_known]],
            preds_labels=[outputs_cls[inputs_cls != self.no_known]],
            preds_connects=[outputs_connects[inputs_connects != self.no_known]],
            preds_coeffs=[outputs_coeffs[inputs_coeffs != self.no_known]]
        )

        loss_inputs = [gt_coords, gt_labels, gt_connects, gt_coeffs, preds_dicts]
        losses = self.pts_bbox_head.loss(*loss_inputs)

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
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
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
        if self.bev_scale != 1.0:
            b, c, h, w = bev_feats.shape
            bev_feats = F.interpolate(bev_feats, (int(h * self.bev_scale), int(w * self.bev_scale)))


        # import cv2
        # bs = bev_feats.shape[0]
        # save_dir = f"vis/pretrain_48x32_bev_feature/"
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # for bi in range(bs):
        #     name = img_metas[bi]['filename'][0].split('/')[-1].split('.jpg')[0]
        #     bev_feat_draw = bev_feats[bi].mean(dim=0)
        #     bev_feat_draw = bev_feat_draw.detach().cpu().numpy()
        #     bev_feat_draw = (bev_feat_draw - bev_feat_draw.min()) / (bev_feat_draw.max() - bev_feat_draw.min())
        #     bev_feat_draw = (bev_feat_draw * 255).astype(np.uint8)
        #     bev_feat_draw = cv2.applyColorMap(bev_feat_draw, cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join(save_dir, f"{name}_bev.png"), bev_feat_draw)
        # import pdb;pdb.set_trace()

        losses = dict()
        gt_lines_coords = [img_meta['centerline_coord'] for img_meta in img_metas]
        gt_lines_labels = [img_meta['centerline_label'] for img_meta in img_metas]
        gt_lines_connects = [img_meta['centerline_connect'] for img_meta in img_metas]
        gt_lines_coeffs = [img_meta['centerline_coeff'] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, gt_lines_coords, gt_lines_labels, 
                                            gt_lines_connects, gt_lines_coeffs,
                                            img_metas, num_coeff, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def vis_linepts(self, pred_points, pred_labels, img_meta, path, aux_name):
        import cv2
        
        # label_color = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        #        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        #        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        #        [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        label_color = [[255, 0, 0], [0, 0, 230], [35, 244, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100],  [119, 11, 32]]
        bev_img = np.zeros((200, 200, 3), np.uint8)
        for i in range(len(pred_points)):
            try:
                bev_img = cv2.circle(bev_img, (int(pred_points[i][0]), int(pred_points[i][1])), 1, label_color[int(pred_labels[i])], 2)
            except:
                pass
        # for i in range(len(pred_points) // 3):
        #     pt1 = (int(pred_points[i * 3][0]), int(pred_points[i * 3][1]))
        #     pt2 = (int(pred_points[i * 3 + 1][0]), int(pred_points[i * 3 + 1][1]))
        #     pt3 = (int(pred_points[i * 3 + 2][0]), int(pred_points[i * 3 + 2][1]))
        #     bev_img = cv2.line(bev_img, pt1, pt2, (0, 255, 0), 1)
        #     bev_img = cv2.line(bev_img, pt2, pt3, (0, 255, 0), 1)
        name = img_meta['filename'][0].split('/')[-1].split('.jpg')[0]
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{name}_{aux_name}.png"), bev_img)
        return
    
    def vis_from_nodelist(self, nodelist, img_meta, path, aux_name):
        Map_size = [(-50, 50), (-50, 50)]
        Map_resolution = 0.5
        image = np.zeros([200, 200, 3])
        point_color_map = {"start": (0, 0, 255), 'fork': (0, 255, 0), "continue": (0, 255, 255), "merge": (255, 0, 0)}

        for idx, node in enumerate(nodelist):
            if node['sque_type'] == 'start':
                cv2.circle(image, node['coord'], 1, color=point_color_map['start'], thickness=2)
            elif node['sque_type'] == 'continue':
                cv2.circle(image, node['coord'], 1, color=point_color_map['continue'], thickness=2)
                # cv2.polylines(image, [subgraphs_points_in_between_nodes[(node.sque_index-1, node.sque_index)]], False, color=point_color_map['continue'], thickness=1)
                cv2.arrowedLine(image, nodelist[idx - 1]['coord'], node['coord'], color=point_color_map['continue'],
                                thickness=1, tipLength=0.1)
            elif node['sque_type'] == 'fork':
                if node['fork_from'] > idx or node['fork_from'] < 0:
                    continue
                cv2.circle(image, node['coord'], 1, color=point_color_map['fork'], thickness=2)
                cv2.arrowedLine(image, nodelist[node['fork_from'] - 1]['coord'], node['coord'],
                                color=point_color_map['fork'],
                                thickness=1, tipLength=0.1)
            elif node['sque_type'] == 'merge':
                if node['merge_with'] > idx or node['merge_with'] < 0:
                    continue
                cv2.circle(image, node['coord'], 1, color=point_color_map['merge'], thickness=2)
                cv2.arrowedLine(image, nodelist[node['merge_with'] - 1]['coord'], node['coord'],
                                color=point_color_map['merge'], thickness=1, tipLength=0.1)

        name = img_meta['filename'][0].split('/')[-1].split('.jpg')[0]
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{name}_{aux_name}.png"), image)

    def simple_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        gt_lines_seqs = [img_meta['centerline_sequence'] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        clause_length = 4 + num_coeff * 2
        # gt_node_list = self.seq2nodelist(gt_lines_seqs[0])
        # self.vis_from_nodelist(gt_node_list, img_metas[0], self.vis_cfg['path'], 'gt')
        device = pts_feats[0].device
        input_seqs = (torch.ones(1, 1).to(device) * self.start).long()
        outs = self.pts_bbox_head(pts_feats, input_seqs, img_metas)
        output_seqs, values = outs
        line_results = []
        for bi in range(output_seqs.shape[0]):
            # gt_node_list = self.seq2nodelist(gt_lines_seqs[bi])
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            if self.end in pred_line_seq:
                stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
            stop_idx = stop_idx // clause_length * clause_length
            pred_line_seq = pred_line_seq[:stop_idx]
            pred_line_seq[2::clause_length] = pred_line_seq[2::clause_length] - self.category_start
            pred_line_seq[3::clause_length] = pred_line_seq[3::clause_length] - self.connect_start
            for k in range(4, clause_length):
                pred_line_seq[k::clause_length] = pred_line_seq[k::clause_length] - self.coeff_start
            pred_node_list = seq2bznodelist(pred_line_seq.detach().cpu().numpy(), n_control, self.epsilon)

            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu(),
                pred_node_lists = pred_node_list
            ))
        return line_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        line_results = self.simple_test_pts(
            bev_feats, img_metas, rescale=rescale)
        for result_dict, line_result in zip(bbox_list, line_results):
            result_dict['line_results'] = line_result
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
