# dataset settings
data_root = 'data/EVStateDetection/Train_data/yolo_train_data_2024_12_18_1/'
# data_root = 'data/ibk/yolo_train_data_2025_2_10_9/train/'
dataset_type = 'CocoDataset'
img_scale = (640, 640)  # width, height
classes = ('Door_open', 'Door_close', 'Door_moving', 'Door_sticker', 'Indicator', 'Hall_LED', 'Hall_LED_on', 'Hall_LED_off', 'up', 'down', 'unknown', 'nine', 'eight', 'seven', 'six', 'five', 'four', 'three', 'two', 'one', 'zero', 'L', 'B', 'P', 'G', 'R', 'F', 'N', 'M')
train_batch_size = 2
train_num_workers = 2
eval_batch_size = 2
eval_num_workers = 2

# pipeline
train_pipeline = [
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
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
                    pad_to_square=True,
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
