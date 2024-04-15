_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py', 
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
transformer_dims=256
transformer_layers=6
head_dims=32

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
    type='LssSegEgo',
    use_grid_mask=True,
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
    data_aug_conf=ida_aug_conf,
    pts_bbox_head=dict(
        type='LssSegHead',
        in_channels=transformer_dims,
        channels=transformer_dims,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),)

dataset_type = 'CustomSegNuScenesDataset'
data_root = './data/nuscenes/'
seg_root = data_root+'centerline_pryseg'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadCenterlineSegFromPkl', grid_conf=grid_conf, thickness=3),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3DSeg', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'center_seg'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'gt_lines_coord', 'gt_lines_label', 'lidar2ego'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='LoadCenterlineSegFromPkl', grid_conf=grid_conf, thickness=3, data_root=seg_root),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
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
                'img_norm_cfg', 'sample_idx', 'timestamp', 'lidar2ego'))
        ])
]

data = dict(
    samples_per_gpu=4,  # 2
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
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'nuscenes_centerline_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'nuscenes_centerline_infos_val.pkl', classes=class_names, modality=input_modality))


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
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline, data_root=seg_root)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'
resume_from=None
