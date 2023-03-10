# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/input/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 720), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('pothole',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes = classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        classes = classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        classes = classes))
evaluation = dict(interval=1, metric='bbox')
