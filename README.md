# Cloud Based Neural Network Inference

## Overview


## Directory Structure
Folders:

	- mqtt:
		- MQTT based inference directory
	- tfserving:
		- TFServing inference directory



├── mqtt
│   ├── bag
│   │   └── README.md
│   │   └── left_camera_templergraben.bag
│   ├── docker
│   │   ├── Dockerfile
│   │   └── run.sh
│   ├── mqtt_node
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   └── src
│   │       ├── cloud_node_no_ros.py
│   │       ├── cloud_node.py
│   │       ├── data
│   │       │   └── image.png
│   │       ├── frozen_graph_runner.py
│   │       ├── launch
│   │       │   ├── cloud_node.launch
│   │       │   ├── run_rosbag.launch
│   │       │   ├── vehicle_node.launch
│   │       │   └── vehicle_node_with_rosbag.launch
│   │       ├── model
│   │       │   ├── best_weights_e=00231_val_loss=0.1518
│   │       │   │   ├── keras_metadata.pb
│   │       │   │   ├── saved_model.pb
│   │       │   │   └── variables
│   │       │   │       ├── variables.data-00000-of-00001
│   │       │   │       └── variables.index
│   │       │   ├── mobilenet_v3_large_968_608_os8.pb
│   │       │   ├── mobilenetv3_large_os8_deeplabv3plus_72miou
│   │       │   │   ├── keras_metadata.pb
│   │       │   │   ├── saved_model.pb
│   │       │   │   └── variables
│   │       │   │       ├── variables.data-00000-of-00001
│   │       │   │       └── variables.index
│   │       │   └── mobilenet_v3_small_968_608_os8.pb
│   │       ├── utils
│   │       │   ├── img_utils.py
│   │       │   └── mqtt_topic_reader.py
│   │       ├── vehicle_node.py
│   │       └── xml
│   │           ├── cityscapes.xml
│   │           └── convert.xml
│   ├── NOTES.md
│   ├── README.md
│   ├── run_broker_encrypted.sh
│   ├── setup_new_ros_ws.sh
│   ├── start_node.sh
│   └── user_setup_docker.sh
├── README.md