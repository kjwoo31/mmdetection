# dataset settings
data_root = 'data/PanoramicViewDetection_seperated/'
dataset_type = 'CocoDataset'
img_scale = (480, 480)  # width, height
# classes = ('person', 'bicycle', 'motorcycle', 'kickboard', 'car', 'bus', 'truck', 'robot', 'animal','unknown')  # class order is wrong
classes = ('person', 'robot', 'car', 'truck', 'motorcycle', 'bicycle', 'bus', 'kickboard', 'animal')
train_batch_size = 16
train_num_workers = 4
eval_batch_size = 1
eval_num_workers = 2

# pipeline
train_pipeline = [
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        pad_to_square=False,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        pad_to_square=False,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=img_scale, keep_ratio=True)
            ],
            [
                dict(
                    type='Pad',
                    size=img_scale,
                    pad_to_square=False,
                    pad_val=dict(img=(114.0, 114.0, 114.0))),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor'))
            ]
        ])
]

# train
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        metainfo=dict(classes=classes),
        data_prefix=dict(img=''),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

# val
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='val.json',
    metainfo=dict(classes=classes),
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=test_pipeline)

val_dataloader = dict(
    batch_size=eval_batch_size,
    num_workers=eval_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False)


# test
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='test.json',
    metainfo=dict(classes=classes),
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=eval_batch_size,
    num_workers=eval_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False)
