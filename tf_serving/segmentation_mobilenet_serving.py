import os
import json
import shutil
import requests
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET



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



#model = tf.keras.models.load_model('model')
#path_to_xml = 'cityscapes.xml'

path_to_xml = 'convert.xml'
color_palette, class_names, color_to_label = parse_convert_xml(path_to_xml)

width = 2048
height = 1024

input_img = cv2.imread('image.png')
input_img = resize_image(input_img,[height,width])



batched_img = tf.expand_dims(input_img, axis=0)
batched_img = tf.cast(batched_img, tf.uint8)
print(f"Batched image shape: {batched_img.shape}")


# model_outputs = model(batched_img)
# print(f"Model output shape: {model_outputs.shape}")
# print(f"Predicted class: {postprocess(model_outputs)}")



### Serving part 

data = json.dumps(
    {"signature_name": "serving_default", "instances": batched_img.numpy().tolist()}
)


# Docker command for setting up Server :
'''
CHECK README
'''
url = "http://localhost:8501/v1/models/mobilenet:predict"


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    #print(response)
    predictions = np.array(response["predictions"])
    prediction = tf.squeeze(predictions).numpy()
    argmax_prediction = np.argmax(prediction, axis=2)
    prediction = segmentation_map_to_rgb(argmax_prediction,color_palette=color_palette).astype(np.uint8)
    #prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)

    return prediction


print("Now using the serving \n")
prediction = predict_rest(data, url)


print(f"REST output shape: {prediction.shape}")
cv2.imshow('prediction',prediction)
cv2.waitKey(0)
#print(f"Predicted class: {postprocess(rest_outputs)}")
