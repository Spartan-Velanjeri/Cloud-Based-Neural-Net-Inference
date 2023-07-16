
import requests
import numpy as np
import tensorflow as tf
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET
import json
import time
import argparse

# model = tf.keras.load_model('model/1')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for semantic segmentation model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the TensorFlow Serving model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    return parser.parse_args()

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


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    #print(response)
    predictions = np.array(response["predictions"])
    # prediction = tf.squeeze(predictions).numpy()
    prediction = np.squeeze(predictions).tolist()  # Convert to list
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction,color_palette=color_palette).astype(np.uint8)
    #prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)

    return prediction

def preprocess(image_path):

    # start = time.time()

    #path_to_xml = 'convert.xml'

    width = 2048
    height = 1024

    input_img = cv2.imread(image_path)
    input_img = resize_image(input_img,[height,width])

    batched_img = np.expand_dims(input_img, axis=0)
    batched_img = batched_img.astype(np.uint8)
    #print(f"Batched image shape: {batched_img.shape}")




def postprocess(response):
    path_to_xml = 'cityscapes.xml'
    color_palette, _,_ = parse_convert_xml(path_to_xml)
    predictions = np.array(response["predictions"])
    # prediction = tf.squeeze(predictions).numpy()
    prediction = np.squeeze(predictions).tolist()  # Convert to list
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction,color_palette=color_palette).astype(np.uint8)