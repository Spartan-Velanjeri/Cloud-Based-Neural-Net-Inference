# vehicle-cloud-inference
Part of the ACDC Research Project 

This branch is mainly for TF-Serving related progress

## TF-Serving Inference method

In order to run the tf-serving

First let's run the Tensorflow Serving Docker Container (CPU) by running

`docker run -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving`


OR Using GPU (CHECK NOTE BEFORE RUNNING) by runnning

`docker run --gpus all -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving:latest-gpu`


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

## Models used:
1. [best_weights_e_00231_val_loss_0.1518.zip](https://git.rwth-aachen.de/ika/acdc-research-project-ss23/acdc-research-project-ss23/uploads/e5bdaf3b7aa6d2b59bbd098e55eb079c/best_weights_e_00231_val_loss_0.1518.zip)
   
2. [mobilenetv3_large_os8_deeplabv3plus_72miou.zip](https://git.rwth-aachen.de/ika/acdc-research-project-ss23/acdc-research-project-ss23/uploads/3f73d5bd57acc307182278c0e0449650/mobilenetv3_large_os8_deeplabv3plus_72miou.zip)

## TO CONNECT WITH THE IKA WORKSTATION (supposed to have tfserving GPU as well though not running)
1. Make sure to connect to the IKA Workstation or any other Cloud station using SSH or the method prescribed to connect. In case of the IKA Workstation, you must turn on the RWTH VPN and then SSH to the Host (See the ACDC RP Presentation for this method)
   
2. Make sure to pull the TFServing Docker Image first (either `docker pull tensorflow/serving` (FOR CPU) or `docker pull tensorflow/serving:latest-gpu` (FOR GPU)). The Workstation(Cloud station) must have the Nvidia Driver and the Nvidia Container Toolkit installed before pulling the docker.
    
3. Run the Docker container on the IKA Workstation using the code (Using the terminal on VSCode which is connected to the Workstation)
   `docker run -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving` (IF CPU)
    
    `docker run --gpus all -p 8501:8501 --mount type=bind,source=/home/typlosion/work/ACDC/vehicle-cloud-inference/model/1/,target=/models/mobilenet/1/ -e MODEL_NAME=mobilenet -t tensorflow/serving:latest-gpu` (IF GPU)

4. Make sure to turn on the port forwarding to 8501 on the port forwardiing tab (usually next to the terminal tab in the bottom window)
5. Now in your program, make sure to change the URL from localhost to that of the server IN the Inference script`segmentation_mobilenet_serving.py`. In case of the IKA Workstation, it is
   `url = "http://i2200049.ika.rwth-aachen.de:8501/v1/models/mobilenet:predict"` instead of `url = "http://localhost:8501/v1/models/mobilenet:predict"`
6. Now just run the `segmentation_mobilenet_serving.py` script and you should be able to view the result on your local system
