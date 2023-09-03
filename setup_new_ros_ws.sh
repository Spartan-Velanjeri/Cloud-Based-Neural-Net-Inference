#!/bin/bash

WS_DIR="."
WS_NAME="ws_mqtt_nodes"

WS_PATH="$WS_DIR/$WS_NAME"

echo "preparing ros ws with package ..."

mkdir -p $WS_PATH/src
echo "create dir: $WS_PATH/src"

cp -r mqtt_node $WS_PATH/src/
echo "copied mqtt_node to newly created workspace folder"

cd $WS_PATH && echo "moved to ros ws folder '${PWD}'. running catkin_make ..."

source /opt/ros/noetic/setup.bash

catkin_make

echo "prepared ros workspace with package."

source devel/setup.bash
