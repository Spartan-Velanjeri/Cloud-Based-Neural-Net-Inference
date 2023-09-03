import json
import requests
import numpy as np
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET
import time
import speedtest
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import tensorflow as tf
import argparse
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
    rgb_encoding = color_palette[segmentation_map]
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
    Make predictions using a remote machine learning model served through a REST API.

    Parameters:
    - json_data: JSON-formatted data containing input for the model.
    - url: The URL endpoint of the TensorFlow Serving server hosting the model.
    - color_palette: A color palette used for post-processing the model's predictions.

    Returns:
    - prediction: The processed prediction result, typically an image or classification output.

    Description:
    This function sends a JSON-formatted input to a remote TensorFlow Serving server specified by the 'url'.
    The server processes the input using the hosted model and returns a prediction response in JSON format.
    The prediction response is then post-processed to obtain the final prediction, often an image or classification result,
    which is returned as the output of this function.

    Note:
    - The function assumes that the TensorFlow Serving server is correctly set up to handle REST API requests.
    - The 'color_palette' parameter is used for converting the model's segmentation map into a colored image.

    Models Available:
    -  best_weights_e=00231_val_loss=0.1518
    -  mobilenetv3_large_os8_deeplabv3plus_72miou

    Example Usage:
    - prediction = predict_rest(json_input, "http://example.com/model_endpoint", color_palette)
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
    """
    Make predictions using gRPC for a machine learning model served by TensorFlow Serving.

    Parameters:
    - data: Input data for prediction.
    - input_name: The name of the input tensor in the model.
    - stub: gRPC stub for communicating with the TensorFlow Serving server.
    - color_palette: A color palette used for post-processing the prediction.

    Returns:
    - prediction: The processed prediction result.

    Description:
    This function sends input data to a TensorFlow Serving server using gRPC for model prediction.
    It assumes the server is hosting a semantic segmentation model.
    The function processes the gRPC response, converts it to an image format, and applies color mapping using the provided color_palette.
    """

    # Create a gRPC request made for prediction
    request = predict_pb2.PredictRequest()

    # Set the name of the model, for this use case it is "model"
    request.model_spec.name = "mobilenet" # Based on the Tensorflow Docker command, under the MODEL_NAME

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
    """
    Convert a byte size to a human-readable format.

    Parameters:
    - nbytes: The size in bytes.

    Returns:
    - human_readable: A string representing the size in a human-friendly format.

    Description:
    This function takes a size in bytes and converts it to a human-readable format (e.g., KB, MB, GB).
    It rounds the size to two decimal places and appends the appropriate unit (bytes, KB, MB, etc.).
    """
    suffixes = ['b', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def bag_reader(bag_file,flag):

    '''
    Reads frames from a ROSBag file and processes them using OpenCV and TensorFlow Serving based on the specified flag.

    Args:
        bag_file (str): The path to the ROSBag file containing camera frames.
        flag (str): A flag indicating whether to use TensorFlow Serving ('grpc') or a REST API ('rest') for inference.
    '''

    while True: # Useful to loop the ROSBag as there ain't existing built-in func
        total_list, prediction_list = [],[]
        avg_total = 0
        avg_prediction = 0
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
                    total,prediction = serving_func(rgb_image,flag)
                    total_list.append(total)
                    prediction_list.append(prediction)
                    #print(len(total_list))
                    if (len(total_list)>5):
                        total_list.pop(0)
                        prediction_list.pop(0)
                        avg_total = sum(total_list)/len(total_list)
                        avg_prediction = sum(prediction_list)/len(prediction_list)
                        print("Average Total Time: ",avg_total,"Average Prediction Time: ",avg_prediction)
                        break
                    
                    
def serving_func(input_img,flag):
    '''
    Processes an input image using gRPC or a REST API based on the specified flag and performs various measurements.

    Args:
        input_img (numpy.ndarray): The input image for inference.
        flag (str): A flag indicating whether to use gRPC ('grpc') or a REST API ('rest') for inference.
    '''
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



    batched_img = np.expand_dims(input_img, axis=0)
    batched_img = batched_img.astype(float)
    #batched_img = batched_img.astype(np.uint8)
    print(f"Batched image shape: {batched_img.shape}")
    preprocess_time = time.time()

    rest_dump = "None"

    if flag == 'rest':
        ### Serving part 
        rest_start = time.time()
        data = json.dumps(
            {"signature_name": "serving_default", "instances": batched_img.tolist()}
        )
        rest_end = time.time()

        # Docker command for setting up Server :
        '''
        CHECK README
        '''
        url = "http://localhost:8501/v1/models/mobilenet:predict" #If tfserving on your local system
        #url = "http://i2200049.ika.rwth-aachen.de:8501/v1/models/mobilenet:predict" # If tfserving on the IKA Workstation





    print("Now using the serving \n")
    pred_start = time.time()
    if flag == 'rest':
        prediction = predict_rest(data, url,color_palette)
        rest_dump = rest_end-rest_start
    else:
        prediction = predict_grpc(batched_img,input_name=input_name,stub=stub,color_palette=color_palette)
    prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
    pred_end = time.time()

    print(f"REST output shape: {prediction.shape}")
    end = time.time()
    total_time = end-start
    prediction_time = pred_end - pred_start
    
    print("total time",total_time)
    print("Preprocessing time", preprocess_time-start)
    print("Prediction time", prediction_time)
    #print("Internet Stats Calc Time", internet_complete-start)
    #print("Ping:",ping)
    #print("Download Speed",ds)
    #print("Upload Speed",us)

    # cv2.imshow('prediction',prediction)
    # cv2.waitKey(1)
    return total_time, prediction_time
#print(f"Predicted class: {postprocess(rest_outputs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify trigger and model_export_path.')
    parser.add_argument('--bag', type=str, default='left_camera_templergraben.bag', help='Path to the Bag file')
    parser.add_argument('--model_export_path', type=str, default='./model/mobilenetv3_large_os8_deeplabv3plus_72miou/', 
                        help='Path to the model export directory. Make sure the model path matches the one TFServing is serving :)')
    parser.add_argument('--trigger', type=str, default='grpc', help='Trigger for mode (e.g., "grpc" or "rest").')
    args = parser.parse_args()
    print("Running the %s Model",args.model_export_path)
    if args.trigger != 'rest': # Anything other than REST API, will trigger the gRPC
        print("Running gRPC mode")
        #gRPC Setting up 
        channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.insecure_channel("0.0.0.0:8500", options=channel_opt) #Change this if using any other system other LocalHost for Cloud
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) # Used to send the gRPC request to the TF Server

        # Get the serving_input key

        model_export_path = args.model_export_path
        loaded_model = tf.saved_model.load(model_export_path)
        input_name = list(
            loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
        )[0]
    else:
        print("Running REST API mode")

    # Bag file 
    bag_reader(bag_file=args.bag,flag=args.trigger)
    

