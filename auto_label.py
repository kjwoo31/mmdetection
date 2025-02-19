from mmdet.apis import DetInferencer
import os

# Parameter
data_path = '/mnt/4_DB_STORED_DATA/ibk_euljiro_head/el_outside/el_state/object_detection/1_original/2_B'
out_dir= '/mnt/4_DB_STORED_DATA/ibk_euljiro_head/el_outside/el_state/object_detection/2_auto_label/2_B'
model_name = 'YOLOX'
score_threshold = 0.5
batch_size = 1
text_prompt = None
custom_entities = False
draw_pred = True  # Whether to draw predicted bounding boxes.
no_save_pred = False  # Whether to force not to save prediction vis results
return_labelme = True  # Whether to return results as labelme format

# Model
if model_name == 'CODETR':
    model_path = 'projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_ev_state.py'
    weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_ev_state/best_coco_bbox_mAP_epoch_32.pth'
elif model_name == 'GroundingDINO': # Need to change text prompt to small letter, alphabet
    model_path = 'configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_ev_state.py'
    weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/grounding_dino_swin-t_finetune_16xb4_1x_ev_state/epoch_16.pth'
    text_prompt = 'Door_open . Door_close . Door_moving . Door_sticker . Indicator . Hall_LED . Hall_LED_on . Hall_LED_off . up . down . unknown . 9 . 8 . 7 . 6 . 5 . 4 . 3 . 2 . 1 . 0 . L . B . P . G . R . F . N . M .'
    custom_entities = True
elif model_name == 'YOLOX':
    model_path = 'configs/yolox/yolox_s_ev_state_detection.py'
    weight_path = '/mnt/DL_TOOLS/mmdetection/work_dirs/yolox_s_ev_state_detection_pretrained_ibk/best_coco_bbox_mAP_epoch_202.pth'

# Initialize the DetInferencer
inferencer = DetInferencer(
    model=model_path,
    weights=weight_path)

# Perform inference
inferencer(
    data_path,
    batch_size=batch_size,
    draw_pred=draw_pred,
    show=False,
    out_dir=out_dir,
    pred_score_thr=0.5,
    no_save_pred=no_save_pred,
    return_labelme=return_labelme,
    texts=text_prompt,
    custom_entities=custom_entities)
