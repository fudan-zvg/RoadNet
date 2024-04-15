_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py', 
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
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
ida_aug_conf = {
        "resize_lim": (0.193, 0.225),
        "final_dim": (128, 352),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='AR_RNTR',
    use_grid_mask=False,
    freeze_pretrain=False,
    img_backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        frozen_stages=0,
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
        out_channels=256,
        num_outs=3),
    lss_cfg=dict(downsample=8, 
                d_in=256, 
                d_out=256),
    grid_conf = grid_conf,
    # vis_cfg=dict(path='val_lss_prycon_l6_256_150'),
    data_aug_conf=ida_aug_conf,
    pts_bbox_head=dict(
        type='ARRNTRHead',
        num_classes=10,
        in_channels=256,
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
            type='LssSeqLineTransformer',
            decoder=dict(
                type='PETRTransformerLineDecoder',
                return_intermediate=True,
                num_layers=transformer_layers,
                transformerlayers=dict(
                    type='PETRLineTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='PETRSelfMultiheadAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1),
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
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=transformer_dims//2, normalize=True),
        bev_positional_encoding=dict(
                     type='PositionEmbeddingSineBEV',
                     num_feats=transformer_dims//2,
                     normalize=True),
        loss_coords=dict(
            type='CrossEntropyLoss'),
        loss_labels=dict(
            type='CrossEntropyLoss', class_weight=label_class_weight),
        loss_connects=dict(
            type='CrossEntropyLoss', class_weight=connect_class_weight), 
        loss_coeffs=dict(
            type='CrossEntropyLoss')
            ))

dataset_type = 'CenterlineNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_centerline_infos_train.pkl',
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
        file_client_args=file_client_args))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='CenterlineFlip', prob=0.5),
    dict(type='CenterlineRotateScale', prob=0.5, max_rotate_degree=22.5, scaling_ratio_range=(0.95, 1.05)),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 
                'centerline_label', 'centerline_connect', 'centerline_coeff', 'centerline_sequence', 
                'lidar2ego', 'n_control'))
        ])
]

data = dict(
    samples_per_gpu=2,  # 2
    workers_per_gpu=4,  # 4
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_centerline_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, 
             pipeline=test_pipeline, 
             ann_file=data_root + 'nuscenes_centerline_infos_val.pkl', 
             classes=class_names, 
             modality=input_modality, 
             grid_conf=grid_conf,
             bz_grid_conf=bz_grid_conf
             ),
    test=dict(type=dataset_type, 
              pipeline=test_pipeline, 
              ann_file=data_root + 'nuscenes_centerline_infos_val.pkl', 
              classes=class_names, 
              modality=input_modality,
              grid_conf=grid_conf,
              bz_grid_conf=bz_grid_conf
              ))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)


optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
total_epochs = 300
evaluation = dict(interval=100, pipeline=test_pipeline, metric='ar_reach')
find_unused_parameters=True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/lss_roadseg_48x32_b4x8_resnet_adam_24e.pth'
resume_from=None
