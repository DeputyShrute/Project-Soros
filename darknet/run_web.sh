#!/bin/bash
gnome-terminal -x sh -c obs --startvirtualcam --minimize-to-tray
sleep 5
./darknet detector demo data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -c 0
