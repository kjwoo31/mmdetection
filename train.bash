#!/bin/bash

# rtmdet
bash tools/dist_train.sh configs/rtmdet/rtmdet_s_rear_view_detection.py 2
bash tools/dist_train.sh configs/rtmdet/rtmdet_s_panoramic_view_detection.py 2
bash tools/dist_train.sh configs/rtmdet/rtmdet_s_traffic_light_detection.py 2

# python
# python tools/test.py configs/yolox/yolox_s_rear_view_detection.py work_dirs/yolox_s_rear_view_detection/epoch_640.pth
# python tools/test.py configs/rtmdet/rtmdet_s_rear_view_detection.py work_dirs/rtmdet_s_rear_view_detection/epoch_1000.pth

# yolox
# bash tools/dist_train.sh configs/yolox/yolox_s_rear_view_detection.py 2
# bash tools/dist_train.sh configs/yolox/yolox_s_panoramic_view_detection.py 2
# bash tools/dist_train.sh configs/yolox/yolox_s_traffic_light_detection.py 2