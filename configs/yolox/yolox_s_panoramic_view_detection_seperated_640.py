_base_ = [
    './yolox_s_compare_20241211.py', '../_base_/datasets/panoramic_view_dataset_seperated_640.py'
]

load_from = './checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa
