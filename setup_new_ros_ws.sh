#!/bin/bash

WS_DIR="."
WS_NAME="ws_mqtt_nodes"

WS_PATH="$WS_DIR/$WS_NAME"
WS_VEHICLE=$WS_PATH/vehicle_node
WS_CLOUD=$WS_PATH/cloud_node

echo "preparing ros ws with package ..."

mkdir -p $WS_CLOUD/src
echo "created dir: $WS_CLOUD/src"

mkdir -p $WS_VEHICLE/src
echo "created dir: $WS_VEHICLE/src"

cp -r mqtt_node $WS_VEHICLE/src/
cp -r mqtt_node $WS_CLOUD/src/
echo "copied mqtt_node to newly created workspace folders for cloud node and vehicle node separately"

echo "folder: $WS_PATH"
ls -l $WS_PATH

echo "copying bag file to vehicle node workspace"
cp bag/left_camera_templergraben.bag $WS_VEHICLE/src/mqtt_node/src/bag
# cd $WS_PATH && echo "moved to ros ws folder '${PWD}'. running catkin_make ..."

source /opt/ros/noetic/setup.bash

# catkin_make

# echo "prepared ros workspace with package."

# source devel/setup.bash
