import json
import requests
import numpy as np
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET
import time
import speedtest
from cv_bridge import CvBridge
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import tensorflow as tf

# Google Remote Procedure Call
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc



def segmentation_map_to_rgb(segmentation_map,color_palette):
    """
    Converts segmentation map to a RGB encoding according to self.color_palette
    Eg. 0 (Class 0) -> Pixel value [128, 64, 128] which is on index 0 of self.color_palette
        1 (Class 1) -> Pixel value [244, 35, 232] which is on index 1 of self.color_palette

    self.color_palette has shape [256, 3]. Each index of the first dimension is associated
    with an RGB value. The index corresponds to the class ID.

    :param segmentation_map: ndarray numpy with shape (height, width)
    :return: RGB encoding with shape (height, width, 3)
    """

    ### START CODE HERE ###
    
    # Task 1:
    # Replace the following command
    #print(segmentation_map)
    rgb_encoding = color_palette[segmentation_map]

    ### END CODE HERE ###
    return rgb_encoding

def parse_convert_xml(conversion_file_path):
    """
    Parse XML conversion file and compute color_palette 
    """

    defRoot = ET.parse(conversion_file_path).getroot()

    color_to_label = {}

    color_palette = np.zeros((256, 3), dtype=np.uint8)
    class_list = np.ones((256), dtype=np.uint8) * 255
    class_names = np.array(["" for _ in range(256)], dtype='<U25')
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        class_name = defElement.get('Name').lower()
        if to_class in class_list:
            color_to_label[tuple(from_color)] = int(to_class)
        else:
            color_palette[idx] = from_color
            class_list[idx] = to_class
            class_names[idx] = class_name
            color_to_label[tuple(from_color)] = int(to_class)

    # Sort classes accoring to is train ID
    sort_indexes = np.argsort(class_list)

    class_list = class_list[sort_indexes]
    class_names = class_names[sort_indexes]
    color_palette = color_palette[sort_indexes]

    return color_palette, class_names, color_to_label

def predict_rest(json_data, url,color_palette):
    
    """
    Models available
    
    best_weights_e=00231_val_loss=0.1518
    mobilenetv3_large_os8_deeplabv3plus_72miou
    
    """
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    #print(response)
    predictions = np.array(response["predictions"])
    print(predictions.shape)
    # prediction = tf.squeeze(predictions).numpy()
    prediction = np.squeeze(predictions).tolist()  # Convert to list
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction,color_palette=color_palette).astype(np.uint8)
    #prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)

    return prediction

def predict_grpc(data, input_name, stub,color_palette):
    # Create a gRPC request made for prediction
    request = predict_pb2.PredictRequest()

    # Set the name of the model, for this use case it is "model"
    request.model_spec.name = "mobilenet" # Based on the Tensorflow Docker 

    # Set which signature is used to format the gRPC query
    # here the default one "serving_default"
    request.model_spec.signature_name = "serving_default"

    # Set the input as the data
    # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(data.tolist()))

    # Send the gRPC request to the TF Server
    result = stub.Predict(request)
    # Process the gRPC response
    output_name = list(result.outputs.keys())[0]
    output_data = result.outputs[output_name].float_val  # Assuming the output is in float format

    # Convert the float data to an image format
    output_data = np.array(output_data)
    print(output_data.shape)
    height = 1024
    width = 2048
    num_channels = 20
    output_data = output_data.reshape((1, height, width, num_channels))  # Adjust the shape accordingly
    #output_data = (output_data * 255).astype(np.uint8)  # Assuming output is in the range [0, 1]

    # Convert the image to RGB
    # print(output_data.shape)
    argmax_prediction = np.argmax(output_data, axis=3)
    prediction = segmentation_map_to_rgb(argmax_prediction, color_palette=color_palette).astype(np.uint8)
    prediction = np.squeeze(prediction)
    # print(prediction.shape)
    return prediction

def humansize(nbytes):
    suffixes = ['b', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])
def bag_reader():
    pass

def main_func(input_img):
    start = time.time()

    #st = speedtest.Speedtest()

    #ds = st.download()
    #ds = humansize(ds)

    #us = st.upload()
    #us = humansize(us)

    #servernames =[]  
    #st.get_servers(servernames)  
    #ping = (st.results.ping)  


    internet_complete = time.time()

    path_to_xml = 'cityscapes.xml'

    #path_to_xml = 'convert.xml'
    color_palette, class_names, color_to_label = parse_convert_xml(path_to_xml)

    width = 2048
    height = 1024

    # input_img = cv2.imread('image.png')
    input_img = resize_image(input_img,[height,width])
    input_img = input_img / 255.0


    # batched_img = tf.expand_dims(input_img, axis=0)
    # batched_img = tf.cast(batched_img, tf.uint8)

    batched_img = np.expand_dims(input_img, axis=0)
    batched_img = batched_img.astype(float)
    #batched_img = batched_img.astype(np.uint8)
    print(f"Batched image shape: {batched_img.shape}")
    preprocess_time = time.time()

    # model_outputs = model(batched_img)
    # print(f"Model output shape: {model_outputs.shape}")
    # print(f"Predicted class: {postprocess(model_outputs)}")



    ### Serving part 

    data = json.dumps(
        {"signature_name": "serving_default", "instances": batched_img.tolist()}
    )


    # Docker command for setting up Server :
    '''
    CHECK README
    '''
    url = "http://localhost:8501/v1/models/mobilenet:predict" #If tfserving on your local system
    #url = "http://i2200049.ika.rwth-aachen.de:8501/v1/models/mobilenet:predict" # If tfserving on the IKA Workstation





    print("Now using the serving \n")
    pred_start = time.time()
    #prediction = predict_rest(data, url,color_palette)
    prediction = predict_grpc(batched_img,input_name=input_name,stub=stub,color_palette=color_palette)
    prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
    pred_end = time.time()

    print(f"REST output shape: {prediction.shape}")
    end = time.time()

    print("total time",end-start)
    print("Preprocessing time", preprocess_time-start)
    print("Prediction time", pred_end - pred_start)
    #print("Internet Stats Calc Time", internet_complete-start)
    #print("Ping:",ping)
    #print("Download Speed",ds)
    #print("Upload Speed",us)

    cv2.imshow('prediction',prediction)
    cv2.waitKey(1)
#print(f"Predicted class: {postprocess(rest_outputs)}")

if __name__ == "__main__":
    #channel = grpc.insecure_channel("0.0.0.0:8500")
    #options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel("0.0.0.0:8500", options=channel_opt)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) # Used to send the gRPC request to the TF Server

    # Get the serving_input key

    model_export_path = './model/mobilenetv3_large_os8_deeplabv3plus_72miou/'
    #model_export_path='./model/best_weights_e=00231_val_loss=0.1518'
    loaded_model = tf.saved_model.load(model_export_path)
    input_name = list(
        loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
    )[0]
    image_file = cv2.imread('image.png')
    bag_file = "left_camera_templergraben.bag"
    with Reader(bag_file) as reader:
    # Iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/sensors/camera/left/image_raw':
                # Assuming 'sensor_msgs/Image' message type
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                
                # Convert ROS image data to OpenCV image
                img_data = msg.data
                #print(img_data)
                width = msg.width
                height = msg.height
                encoding = msg.encoding
                np_arr = np.frombuffer(img_data, np.uint8)
                reshaped_array = np_arr.reshape((height, width))  # Assuming 3 channels (BGR in OpenCV)
                reshaped_array = ((reshaped_array - reshaped_array.min()) / (reshaped_array.max() - reshaped_array.min()) * 255).astype(np.uint8)
                demosaiced_image = cv2.cvtColor(reshaped_array, cv2.COLOR_BAYER_RG2BGR) # Required to convert BAYER format to BGR
                rgb_image = cv2.cvtColor(demosaiced_image, cv2.COLOR_BGR2RGB) #BGR to RGB
                cv2.imshow("input_image",rgb_image)
                cv2.waitKey(1)
                main_func(input_img=rgb_image)