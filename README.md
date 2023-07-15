# vehicle-cloud-inference
Part of the ACDC Research Project 

This branch is mainly for TF-Serving related progress

## TF-Serving Inference method

In order to run the tf-serving

First let's run the Tensorflow Serving Docker Container (CPU) by running

`docker run -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/tf_serving/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving`


OR Using GPU (CHECK NOTE BEFORE RUNNING) by runnning

`docker run --gpus all -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/tf_serving/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving:latest-gpu`


After that, just run the segmentation_mobilenet_serving.py to get the inference

NOTE
1. In order to run the tensorflow serving docker, you must pull them first

FOR CPU
`docker pull tensorflow/serving` 

FOR GPU 

You've gotta to install the nvidia driver and the nvidia container toolkit before using this

check out the links below
[Nvidia Driver installation](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04)

[Nvidia Container toolkit installation for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker)

Once set up,

`docker pull tensorflow/serving:latest-gpu` 

Models used:
1. best_weights_e_00231_val_loss_0.1518.zip
2. mobilenetv3_large_os8_deeplabv3plus_72miou.zip
