#!/bin/bash

docker run \
-it --rm --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--gpus all \
--shm-size=8g \
-v /home/robotis/:/home/robotis \
-v /mnt/:/mnt/ \
mmdetection