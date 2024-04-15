import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import ConvModule
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
import numpy as np
import math
from mmseg.models.builder import build_loss
from mmcv.runner import BaseModule
import copy


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class RegLayer(nn.Module):
    def __init__(self, embed_dims=256,
                 shared_reg_fcs=2,
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                Linear(embed_dims, embed_dims),
                act_layer(),
                Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)

        return embeddings


@HEADS.register_module()
class ARRNTRHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_center_classes,
                 max_center_len=601,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 bev_positional_encoding=dict(
                     type='PositionEmbeddingSineBEV',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_coords=dict(
                     type='CrossEntropyLoss', ),
                 loss_labels=dict(
                     type='CrossEntropyLoss', ),
                 loss_connects=dict(
                     type='CrossEntropyLoss', ),
                 loss_coeffs=dict(
                     type='CrossEntropyLoss', ),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_level=0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 init_cfg=None,
                 normedlinear=False,
                 with_fpe=False,
                 with_time=False,
                 with_multi=False,
                 **kwargs): 
        super(ARRNTRHead, self).__init__(init_cfg=init_cfg)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.max_iteration = max_center_len - 1
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_fpe = with_fpe
        self.with_time = with_time
        self.with_multi = with_multi
        self.group_reg_dims = group_reg_dims
        

        self.loss_coords = build_loss(loss_coords)
        self.loss_labels = build_loss(loss_labels)
        self.loss_connects = build_loss(loss_connects)
        self.loss_coeffs = build_loss(loss_coeffs)

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        if self.in_channels != self.embed_dims:
            self.bev_proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
            
        self.embedding = DecoderEmbeddings(num_center_classes, self.embed_dims, max_center_len)
        self.vocab_embed = MLP(self.embed_dims, self.embed_dims, num_center_classes, 3)
        self._init_layers()
        self.bev_position_encoding = build_positional_encoding(
            bev_positional_encoding)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_fpe:
            self.fpe = SELayer(self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                    self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                    self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                    self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask


    # @auto_fp16(apply_to=('mlvl_feats', 'input_seqs'), out_fp32=True)
    def forward(self, mlvl_feats, input_seqs, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        x = mlvl_feats  # [1, 256, 200, 200]
        if self.in_channels != self.embed_dims:
            x = self.bev_proj(x)
        pos_embed = self.bev_position_encoding(x)  # [1, 256, 200, 200]
        B, _, H, W = x.shape
        masks = torch.zeros(B, H, W).bool().to(x.device)  # [1, 200, 200]

        if self.training:
            tgt = self.embedding(input_seqs.long())  # [1, 301, 256]
            query_embed = self.embedding.position_embeddings.weight  # [301, 256]
            outs_dec, _ = self.transformer(tgt, x, masks, query_embed, pos_embed)
            outs_dec = torch.nan_to_num(outs_dec)
            out = self.vocab_embed(outs_dec)  # [6, 2, 301, 2003]
            return out
        else:
            values = []
            for _ in range(self.max_iteration):
                tgt = self.embedding(input_seqs.long())
                query_embed = self.embedding.position_embeddings.weight

                outs_dec, _ = self.transformer(tgt, x, masks, query_embed, pos_embed)
                outs_dec = torch.nan_to_num(outs_dec)[-1, :, -1, :]
                out = self.vocab_embed(outs_dec)  # [6, 2, 301, 2003]
                out = out.softmax(-1)
                value, extra_seq = out.topk(dim=-1, k=1)
                input_seqs = torch.cat([input_seqs, extra_seq], dim=-1)
                values.append(value)
            values = torch.cat(values, dim=-1)
            return input_seqs, values

    def loss_single(self,
                    preds_coords,
                    preds_labels,
                    preds_connects,
                    preds_coeffs,
                    gt_coords,
                    gt_labels,
                    gt_connects,
                    gt_coeffs,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        loss_coords = self.loss_coords(preds_coords, gt_coords)
        loss_labels = self.loss_labels(preds_labels, gt_labels)
        loss_connects = self.loss_connects(preds_connects, gt_connects)
        loss_coeffs = self.loss_coeffs(preds_coeffs, gt_coeffs)

        loss_coords = torch.nan_to_num(loss_coords)
        loss_labels = torch.nan_to_num(loss_labels)
        loss_connects = torch.nan_to_num(loss_connects)
        loss_coeffs = torch.nan_to_num(loss_coeffs)
        return loss_coords, loss_labels, loss_connects, loss_coeffs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_coords_list,
             gt_labels_list,
             gt_connects_list,
             gt_coeffs_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        preds_coords = preds_dicts['preds_coords']
        preds_labels = preds_dicts['preds_labels']
        preds_connects = preds_dicts['preds_connects']
        preds_coeffs = preds_dicts['preds_coeffs']

        loss_coords, loss_labels, loss_connects, loss_coeffs = multi_apply(
            self.loss_single,
            preds_coords,
            preds_labels,
            preds_connects,
            preds_coeffs,
            gt_coords_list,
            gt_labels_list,
            gt_connects_list,
            gt_coeffs_list,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_coords'] = loss_coords
        loss_dict['loss_labels'] = loss_labels
        loss_dict['loss_connects'] = loss_connects
        loss_dict['loss_coeffs'] = loss_coeffs
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
