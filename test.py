from mmdet.apis import DetInferencer
import os

model = 'rtmdet'
mode = 'trafficlight'
if model == 'yolox':
    if mode == 'panoramic':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/yolox/yolox_s_panoramic_view_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/yolox_s_panoramic_view_detection/best_coco_bbox_mAP_epoch_370.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1708336342548891710.jpg'
    elif mode == 'rear':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/yolox/yolox_s_rear_view_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/yolox_s_rear_view_detection/best_coco_bbox_mAP_epoch_253.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1695344010946721773_und_multi.jpg'
    elif mode == 'trafficlight':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/yolox/yolox_s_traffic_light_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/yolox_s_traffic_light_detection/best_coco_bbox_mAP_epoch_368.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1656919544732607669.jpg'
elif model == 'rtmdet':
    if mode == 'panoramic':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/rtmdet/rtmdet_s_panoramic_view_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/rtmdet_s_panoramic_view_detection/best_coco_bbox_mAP_epoch_200.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1708336342548891710.jpg'
    elif mode == 'rear':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/rtmdet/rtmdet_s_rear_view_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/rtmdet_s_rear_view_detection/best_coco_bbox_mAP_epoch_80.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1695344010946721773_und_multi.jpg'
    elif mode == 'trafficlight':
        model_path = '/mnt/DL_TOOLS/mmdetection/configs/rtmdet/rtmdet_s_traffic_light_detection.py'
        weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/rtmdet_s_traffic_light_detection/best_coco_bbox_mAP_epoch_90.pth'
        file_path = '/mnt/DL_TOOLS/mmdetection/data/test/1656919544732607669.jpg'

# Initialize the DetInferencer
inferencer = DetInferencer(model=model_path, weights=weight_path)

# Perform inference
inferencer(file_path, show=False, out_dir=os.path.dirname(file_path), pred_score_thr=0.5)
