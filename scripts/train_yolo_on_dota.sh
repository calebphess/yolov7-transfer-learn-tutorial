#!/bin/bash

cd yolov7
python train.py --workers 8 --batch-size 32 --data ../config/DOTA-example/dota_data.yaml --img 640 640 --cfg ../config/DOTA-example/dota_cfg.yaml --weights '../models/yolov7-tiny.pt' --name yolov7-overhead --hyp ../config/DOTA-example/dota_hyp.scratch.yaml