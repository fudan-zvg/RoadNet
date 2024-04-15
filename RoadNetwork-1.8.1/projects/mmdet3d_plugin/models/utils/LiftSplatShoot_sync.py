import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner.base_module import BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import xavier_init, constant_init
from torchvision.models.resnet import resnet18
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet3d.models import builder


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 norm_cfg):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, norm_cfg=None):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 norm_cfg=norm_cfg)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 norm_cfg=norm_cfg)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 norm_cfg=norm_cfg)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 norm_cfg=norm_cfg)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=None):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

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



def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]).floor()
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        # save kept for backward
        ctx.save_for_backward(kept)
        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


class CamEncode(BaseModule):
    def __init__(self, depth, d_in=256, d_out=256):
        super(CamEncode, self).__init__()
        self.depth = depth
        self.d_in = d_in
        self.d_out = d_out
        self.depthnet = nn.Conv2d(self.d_in, self.depth + self.d_out, kernel_size=1, padding=0)

    def init_weights(self):
        """Initialize the depthnet weights."""
        xavier_init(self.depthnet, distribution='uniform', bias=0.)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.depth])
        new_x = depth.unsqueeze(1) * x[:, self.depth:(self.depth + self.d_out)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x


class DepthSupCamEncode(BaseModule):
    def __init__(self, depth, d_in=256, d_out=256, d_mid=256, norm_cfg=None):
        super(DepthSupCamEncode, self).__init__()
        self.depth = depth
        self.d_in = d_in
        self.d_out = d_out
        self.d_mid = d_mid
        self.bn = build_norm_layer(norm_cfg, 13)[1]
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(d_in,
                      d_mid,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            build_norm_layer(norm_cfg, d_mid)[1], 
            nn.ReLU(inplace=True),
        )
        self.context_se = SELayer(d_mid)  # NOTE: add camera-aware
        self.depth_se = SELayer(d_mid)  # NOTE: add camera-aware
        self.context_mlp = Mlp(13, d_mid, d_mid)
        self.depth_mlp = Mlp(13, d_mid, d_mid)
        self.context_conv = nn.Conv2d(d_mid,
                                      d_out,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.depth_conv = nn.Sequential(
            BasicBlock(d_mid, d_mid, norm_cfg=norm_cfg),
            BasicBlock(d_mid, d_mid, norm_cfg=norm_cfg),
            BasicBlock(d_mid, d_mid, norm_cfg=norm_cfg),
            ASPP(d_mid, d_mid, norm_cfg=norm_cfg),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=d_mid,
                out_channels=d_mid,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(d_mid,
                      depth,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    # def init_weights(self):
    #     """Initialize the depthnet weights."""
    #     xavier_init(self.depthnet, distribution='uniform', bias=0.)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x, mlp_input):
        # Depth
        mlp_input = self.bn(mlp_input)
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)

        new_x = depth.softmax(dim=1).unsqueeze(1) * context.unsqueeze(2)
        return depth, new_x

    def forward(self, x, mlp_input):
        depth, x = self.get_depth_feat(x, mlp_input)
        return depth, x


class BevEncode(nn.Module):
    def __init__(self, inC, outC, norm_cfg=None):
        super(BevEncode, self).__init__()
        resnet18_cfg = dict(type='ResNet',
                            in_channels=inC,
                            depth=18,
                            num_stages=3,
                            out_indices=(0, 1, 2),
                            strides=(1, 2, 2),
                            dilations=(1, 1, 1),
                            norm_cfg=norm_cfg)
        self.trunk = builder.build_backbone(resnet18_cfg)
        # self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = build_norm_layer(norm_cfg, 64, postfix=1)
        # self.relu = nn.ReLU(inplace=True)
        # block, stage_blocks = BasicBlock, (2, 2, 2, 2)
        self.up1 = Up(64 + 256, 256, scale_factor=4, norm_cfg=norm_cfg)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.trunk.conv1(x)
        x = self.trunk.norm1(x)
        x = self.trunk.relu(x)
        outs = []
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                outs.append(x)
        x = self.up1(outs[2], outs[0])
        x = self.up2(x)
        return x


class LiftSplatShootEgoSyncBN(BaseModule):
    def __init__(self, grid_conf, data_aug_conf, downsample, d_in, d_out, return_bev=False, norm_cfg=None):
        super(LiftSplatShootEgoSyncBN, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = downsample
        self.d_in = d_in
        self.d_out = d_out  # 输出通道数
        self.frustum = self.create_frustum()
        self.depth, _, _, _ = self.frustum.shape  # D是depth
        self.camencode = CamEncode(self.depth, self.d_in, self.d_out)  # 深度，输出通道

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        self.return_bev = return_bev
        self.pc_range = torch.cat((self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))
        self.bevencode = BevEncode(inC=d_out, outC=d_out, norm_cfg=norm_cfg)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = math.ceil(ogfH / self.downsample), math.ceil(ogfW / self.downsample)  # og = original, f = feature
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH,
                                                                                              fW)  # dbound=[4.0, 45.0, 1.0],
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)  # 这里的H和W实际上是原图中的H和W

    def get_geometry(self, img_metas):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        frustum = self.frustum + 0

        # cam_intrinsic = []
        # lidar2cam = []
        # for img_meta in img_metas:
        #     # cam_intrinsic.append(img_meta['cam_intrinsic'])
        #     cam_intrinsic.append(img_meta['intrinsics'])
        #     lidar2cam.append(img_meta['extrinsics'])
        
        D, H, W, _ = frustum.shape
        img2egos = []
        for img_meta in img_metas:
            img2ego = []
            for i in range(len(img_meta['lidar2img'])):
                img2ego.append(img_meta['lidar2ego'] @ np.linalg.inv(img_meta['lidar2img'][i]))
            img2egos.append(np.asarray(img2ego))
        img2egos = np.asarray(img2egos)
        img2egos = frustum.new_tensor(img2egos) # (B, N, 4, 4)
        B, N = img2egos.shape[:2]
        img2egos = img2egos.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, D, H, W, 1, 1)

        # cam_intrinsic = np.asarray(cam_intrinsic)
        # lidar2cam = np.asarray(lidar2cam)
        # cam_intrinsic = frustum.new_tensor(cam_intrinsic)
        # lidar2cam = frustum.new_tensor(lidar2cam)
        # B, N = cam_intrinsic.shape[:2]
        frustum[..., 0:2] *= frustum[..., 2:3]
        points = torch.cat((frustum[..., 0:2], frustum[..., 2:3], torch.ones_like(frustum[..., 0:1])), dim=-1)
        points = points.repeat(B, N, 1, 1, 1, 1)
        # points_cam = torch.inverse(cam_intrinsic).view(B, N, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1))
        # points_lidar = torch.inverse(lidar2cam).view(B, N, 1, 1, 1, 4, 4).matmul(points_cam).squeeze(-1)[..., :-1]
        points_ego = img2egos.matmul(points.unsqueeze(-1)).squeeze(-1)[..., :-1]
        return points_ego

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        x = self.camencode(x.view(B * N, C, imH, imW))
        # 出来一个单层带深度特征图 B*N, dim, depth, imH, imW
        x = x.view(B, N, self.d_out, self.depth, imH, imW)  # [6, 256, 41, 8, 22]
        x = x.permute(0, 1, 3, 4, 5, 2)  # [1, 6, 41, 8, 22, 256]
        # B, N, depth, imH, imW, C
        return x

    def voxel_pooling(self, geom_feats, x, img_metas):

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)  # [43296, 256]

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # [43296, 4]

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        #
        # # BEV-sapce Data Augmentation
        # geom_feats = geom_feats.view(B, -1, 3)
        # # for i in range(B):
        # #     img_meta = img_metas[i]
        # #     for aug in img_meta['transformation_3d_flow']:
        # #         geom_feats[i] = pts_aug(geom_feats[i], aug, img_meta)
        # # flatten indices
        # geom_feats = geom_feats.view(Nprime, 3)
        # # if self.return_bev:
        # #     coors = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        # # batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
        # #                                  device=x.device, dtype=torch.long) for ix in range(B)])
        # # geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # # if self.return_bev:
        # #     coors = torch.cat((coors, batch_ix), 1)
        #
        # # filter out points that are outside box
        # kept = (geom_feats[:, 0] >= self.pc_range[0]) & (geom_feats[:, 0] < self.pc_range[3]) \
        #        & (geom_feats[:, 1] >= self.pc_range[1]) & (geom_feats[:, 1] < self.pc_range[4]) \
        #        & (geom_feats[:, 2] >= self.pc_range[2]) & (geom_feats[:, 2] < self.pc_range[5])
        x = x[kept]  # ([43154, 256])
        geom_feats = geom_feats[kept]  # ([43154, 4])

        # if self.return_bev:
            # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # cumsum trick
        if not self.use_quickcumsum:
            x, coors = cumsum_trick(x, geom_feats, ranks)
        else:
            x, coors = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, int(self.nx[2]), int(self.nx[1]), int(self.nx[0])), device=x.device)
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        final[coors[:, 3], :, coors[:, 2], coors[:, 1], coors[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def get_voxels(self, x, img_metas):  # x得是个两层特征图
        geom = self.get_geometry(img_metas)  # B x N x D x H/downsample x W/downsample x 3

        # import cv2
        # import os
        # save_dir = f"vis/lss_mono_geom/"
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # tmp_geom = geom[:,0,...].reshape(-1, 3).detach().cpu().numpy()
        # pc_range = self.pc_range.cpu().numpy()
        # inbev_x = np.logical_and(tmp_geom[:,0] < pc_range[3], tmp_geom[:,0] >= pc_range[0])
        # inbev_y = np.logical_and(tmp_geom[:,1] < pc_range[4], tmp_geom[:,1] >= pc_range[1])
        # inbev_xy = np.logical_and(inbev_x, inbev_y)
        # tmp_geom = tmp_geom[inbev_xy, :]
        # tmp_xy = np.round(tmp_geom-np.array(pc_range[0:3]))
        # img_draw = np.zeros((int(pc_range[4]-pc_range[1]), int(pc_range[3]-pc_range[0]), 3)).astype(np.uint8)
        # num_sample = tmp_xy.shape[0]
        # radius = 3
        # # Blue color in BGR
        # color = (255, 0, 0)
        # # Line thickness of 2 px
        # thickness = -1
        # for i in range(num_sample):
            
        #     img_draw = cv2.circle(img_draw, (int(tmp_xy[i, 0]), int(tmp_xy[i, 1])), radius, color, thickness)
        # cv2.imwrite(os.path.join(save_dir, f"geom.png"), img_draw)
        # import pdb;pdb.set_trace()

        x = self.get_cam_feats(x)  # B, N, depth, H/downsample x W/downsample, dim(channels)
        x = self.voxel_pooling(geom, x, img_metas)
        return x

    def forward(self, x, img_metas):
        x = self.get_voxels(x, img_metas)
        x = self.bevencode(x)
        # # visualize bev_feature
        # import cv2
        # import numpy as np
        # bev_feature_np = abs(x.mean(dim=1)[0].detach().cpu().numpy())
        # bev_feature_np = (bev_feature_np - 0) / (bev_feature_np.mean()*2 - 0.0) * 255
        # bev_feature_np = bev_feature_np.astype(np.uint8)
        # bev_feature_np = cv2.applyColorMap(bev_feature_np, cv2.COLORMAP_JET)
        # cv2.imwrite('bev_feature.png', bev_feature_np)
        # pdb.set_trace()
        return x

