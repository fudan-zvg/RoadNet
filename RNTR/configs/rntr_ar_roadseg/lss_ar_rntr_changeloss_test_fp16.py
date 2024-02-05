_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/default_runtime.py',
    '../../../../configs/_base_/schedules/cyclic-20e.py'
]

custom_imports = dict(imports=['projects.RNTR.rntr'])
randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)

transformer_dims=256
transformer_layers=6
head_dims=32
num_center_classes = 576
label_class_weight = [1.0] * num_center_classes
connect_class_weight = [1.0] * num_center_classes
label_class_weight[201] = 0.2
connect_class_weight[250] = 0.2

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
grid_conf = dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0],)
bz_grid_conf = dict(
        xbound=[-55.0, 55.0, 0.5],
        ybound=[-55.0, 55.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0],)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_camera=True, use_lidar=False)
ida_aug_conf = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

model = dict(
    type='LssPryBzSeqEgoChangeloss',
    use_grid_mask=False,
    freeze_pretrain=False,
    img_backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[512, 1024, 2048],
        out_channels=transformer_dims,
        num_outs=3),
    lss_cfg=dict(downsample=8, 
                d_in=transformer_dims, 
                d_out=transformer_dims),
    grid_conf = grid_conf,
    # vis_cfg=dict(path='val_lss_prycon_l6_256_150'),
    data_aug_conf=ida_aug_conf,
    pts_bbox_head=dict(
        type='LssPryBzSeqHead',
        num_classes=10,
        in_channels=transformer_dims,
        max_center_len=601,
        num_center_classes=num_center_classes,
        embed_dims=transformer_dims,
        num_query=900,
        LID=True,
        with_position=False,
        with_multiview=False,
        with_fpe=False,
        with_time=False,
        with_multi=False,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='LssSeqLineFlashTransformer',
            decoder=dict(
                type='PETRTransformerLineDecoder',
                return_intermediate=True,
                num_layers=transformer_layers,
                transformerlayers=dict(
                    type='RNTRLineFlashTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='RNTRMultiheadFlashAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1, 
                            batch_first=True,
                            causal=True,
                            ),
                        dict(
                            type='RNTRMultiheadFlashAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1, 
                            batch_first=True,
                            causal=False,
                            ),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=transformer_dims,
                        feedforward_channels=transformer_dims * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=transformer_dims//2, normalize=True),
        bev_positional_encoding=dict(
                     type='PositionEmbeddingSineBEV',
                     num_feats=transformer_dims//2,
                     normalize=True),
        loss_coords=dict(
            type='mmdet.CrossEntropyLoss'),
        loss_labels=dict(
            type='mmdet.CrossEntropyLoss', class_weight=label_class_weight),
        loss_connects=dict(
            type='mmdet.CrossEntropyLoss', class_weight=connect_class_weight), 
        loss_coeffs=dict(
            type='mmdet.CrossEntropyLoss')
            ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4)))

dataset_type = 'CenterlineNuScenesDataset'
data_root = './data/nuscenes/'

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_prycenterline_pon_split_infos_train_v2.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusClearOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf, clear=True),
    dict(type='CenterlineFlip', prob=0.5),
    dict(type='CenterlineRotateScale', prob=0.5, max_rotate_degree=22.5, scaling_ratio_range=(0.95, 1.05)),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='Pack3DDetInputs', 
         keys=['img'], 
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control')),
]
test_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusClearOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf, clear=True),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='Pack3DDetInputs', keys=['img'], 
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,  # 4
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_prycenterline_pon_split_infos_train_v2.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        test_mode=False,
        use_valid_flag=True,
        metainfo=metainfo,
        box_type_3d='LiDAR', 
        backend_args=backend_args,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf,
        ))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        ann_file='nuscenes_prycenterline_pon_split_infos_val_v2.pkl', 
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        modality=input_modality, 
        metainfo=metainfo,
        test_mode=True,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        ann_file='nuscenes_prycenterline_pon_split_infos_val_v2.pkl', 
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        modality=input_modality, 
        metainfo=metainfo,
        test_mode=True,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf))

optim_wrapper = dict(
    # TODO Add Amp
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

num_epochs = 300

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )


train_cfg = dict(max_epochs=num_epochs, val_interval=300)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

load_from='ckpts/lssego_roadseg_48x32_b4x8_resnet_adam_24e_ponsplit_aug_24.pth'
resume = False

# mAP: 0.4104
# mATE: 0.7226
# mASE: 0.2692
# mAOE: 0.4529
# mAVE: 0.3893
# mAAE: 0.1933
# NDS: 0.5025
# Eval time: 206.1s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.581   0.536   0.149   0.076   0.347   0.190
# truck   0.371   0.748   0.205   0.093   0.341   0.216
# bus     0.442   0.703   0.204   0.097   0.758   0.256
# trailer 0.231   1.031   0.237   0.690   0.270   0.136
# construction_vehicle    0.129   1.064   0.494   1.175   0.138   0.356
# pedestrian      0.485   0.676   0.293   0.535   0.443   0.186
# motorcycle      0.407   0.663   0.255   0.579   0.569   0.190
# bicycle 0.416   0.605   0.250   0.689   0.248   0.018
# traffic_cone    0.555   0.545   0.321   nan     nan     nan
# barrier 0.487   0.655   0.284   0.143   nan     nan
