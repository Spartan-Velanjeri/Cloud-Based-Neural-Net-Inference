import json
import requests
import numpy as np
from img_utils import resize_image
import cv2
import xml.etree.ElementTree as ET
import time
import speedtest

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

def predict_rest(json_data, url):
    
    """
    Models available
    
    best_weights_e=00231_val_loss=0.1518
    mobilenetv3_large_os8_deeplabv3plus_72miou
    
    """
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


def humansize(nbytes):
    suffixes = ['b', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


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

input_img = cv2.imread('image.png')
input_img = resize_image(input_img,[height,width])
input_img = input_img / 255.0


# batched_img = tf.expand_dims(input_img, axis=0)
# batched_img = tf.cast(batched_img, tf.uint8)

batched_img = np.expand_dims(input_img, axis=0)
batched_img = batched_img.astype(np.float)
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
prediction = predict_rest(data, url)
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
cv2.waitKey(0)
#print(f"Predicted class: {postprocess(rest_outputs)}")

