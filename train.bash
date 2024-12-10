#!/bin/bash

# max_retries=50
# retry_count=0

# while true; do
#   if ! bash tools/dist_train.sh configs/rtmdet/rtmdet_s_rear_view_detection.py 2 --resume; then
#     ((retry_count++))
#     if [[ $retry_count -gt $max_retries ]]; then
#       echo "Maximum retries reached. Exiting."
#       exit 1
#     fi
#     echo "Retrying..."
#     sleep 1  # Adjust the sleep time as needed
#   else
#     break
#   fi
# done

# while true; do
#   if ! python tools/train.py configs/yolox/yolox_s_rear_view_detection.py --auto-scale-lr; then
#     ((retry_count++))
#     if [[ $retry_count -gt $max_retries ]]; then
#       echo "Maximum retries reached. Exiting."
#       exit 1
#     fi
#     echo "Retrying..."
#     sleep 1  # Adjust the sleep time as needed
#   else
#     break
#   fi
# done

# while true; do
#   if ! python tools/train.py configs/yolox/yolox_s_panoramic_view_detection.py --auto-scale-lr; then
#     ((retry_count++))
#     if [[ $retry_count -gt $max_retries ]]; then
#       echo "Maximum retries reached. Exiting."
#       exit 1
#     fi
#     echo "Retrying..."
#     sleep 1  # Adjust the sleep time as needed
#   else
#     break
#   fi
# done

# while true; do
#   if ! python tools/train.py configs/yolox/yolox_s_traffic_light_detection.py --auto-scale-lr; then
#     ((retry_count++))
#     if [[ $retry_count -gt $max_retries ]]; then
#       echo "Maximum retries reached. Exiting."
#       exit 1
#     fi
#     echo "Retrying..."
#     sleep 1  # Adjust the sleep time as needed
#   else
#     break
#   fi
# done

# python tools/test.py configs/yolox/yolox_s_rear_view_detection.py work_dirs/yolox_s_rear_view_detection/epoch_640.pth

# python tools/test.py configs/rtmdet/rtmdet_s_rear_view_detection.py work_dirs/rtmdet_s_rear_view_detection/epoch_1000.pth


# bash tools/dist_train.sh configs/rtmdet/rtmdet_s_rear_view_detection.py 2 --resume
bash tools/dist_train.sh configs/yolox/yolox_s_rear_view_detection.py 2 --resume
bash tools/dist_train.sh configs/yolox/yolox_s_panoramic_view_detection.py 2 --resume
bash tools/dist_train.sh configs/yolox/yolox_s_traffic_light_detection.py 2 --resume