#!/bin/bash

cd yolov7
python detect.py --weights ../models/yolov7-tiny.pt --conf 0.25 --img-size 640 --source 0
