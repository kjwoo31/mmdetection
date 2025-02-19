
_base_ = [
    './test_model.py', '../_base_/datasets/ev_state_dataset.py'
]

load_from = './work_dirs/yolox_s_ev_state_detection_pretrained/best_coco_bbox_mAP_epoch_161.pth'  # noqa
