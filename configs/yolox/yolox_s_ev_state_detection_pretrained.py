_base_ = [
    './yolox_s_ev_state_detection_model.py', '../_base_/datasets/ev_state_dataset.py'
]

load_from = './checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa
