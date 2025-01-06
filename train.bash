#!/bin/bash

# Auto-label
pip install fairscale transformers
# Grounding DINO
# bash tools/dist_train.sh configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_ev_state.py 2 --resume
# CO-DETR
bash tools/dist_train.sh projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_ev_state.py 2 --resume

# rtmdet
# bash tools/dist_train.sh configs/rtmdet/rtmdet_s_rear_view_detection.py 2
# bash tools/dist_train.sh configs/rtmdet/rtmdet_s_panoramic_view_detection.py 2
# bash tools/dist_train.sh configs/rtmdet/rtmdet_s_traffic_light_detection.py 2

# python
# python tools/test.py configs/yolox/yolox_s_rear_view_detection.py work_dirs/yolox_s_rear_view_detection/epoch_640.pth
# python tools/test.py configs/rtmdet/rtmdet_s_rear_view_detection.py work_dirs/rtmdet_s_rear_view_detection/epoch_1000.pth

# yolox
# bash tools/dist_train.sh configs/yolox/yolox_s_rear_view_detection.py 2
# bash tools/dist_train.sh configs/yolox/yolox_s_panoramic_view_detection.py 2
# bash tools/dist_train.sh configs/yolox/yolox_s_traffic_light_detection.py 2