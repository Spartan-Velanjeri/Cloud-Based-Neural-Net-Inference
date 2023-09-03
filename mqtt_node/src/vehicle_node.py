#!/usr/bin/python3


import rospy
from sensor_msgs.msg import Image
import paho.mqtt.client as mqtt
from PIL import Image as PilImage
import numpy as np
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt


# MQTT Broker
MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_PUB_CAMERA_TOPIC = "/vehicle_camera"
MQTT_SUB_SEGMENTED_TOPIC = "/segmented_images"


# ROS
ROS_PUB_SEGMENTED_TOPIC = "/segmented_ros_images"
ROS_SUB_CAMERA_TOPIC = "/sensors/camera/left/image_raw"

# class ImageSubscriber:
#     def __init__(self):
#         self.bridge = CvBridge()

#         # MQTT setup
#         self.mqtt_client = mqtt.Client()
#         self.mqtt_client.on_message = self.on_mqtt_message
#         self.mqtt_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)
#         self.mqtt_client.subscribe(MQTT_SUB_SEGMENTED_TOPIC)
#         self.mqtt_client.loop_start()

#         # ROS setup
#         self.ros_pub = rospy.Publisher(ROS_PUB_SEGMENTED_TOPIC, Image, queue_size=10)
#         rospy.init_node('mqtt_image_subscriber', anonymous=True)

#     def on_mqtt_message(self, client, userdata, msg):
#             rospy.loginfo_once("Received segmented image from MQTT")
#             np_arr = np.frombuffer(msg.payload, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Seems to work with Jpg format
            
#             # Display image using matplotlib
#             plt.imshow(cv_image)
#             plt.show()

#             # Publish image to ROS topic
#             ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
#             self.ros_pub.publish(ros_image)



# class ImagePublisher:
#     def __init__(self):
#         self.bridge = CvBridge()
        
#         # MQTT setup
#         self.mqtt_client = mqtt.Client()
#         #self.mqtt_client.on_message = self.on_mqtt_message
#         self.mqtt_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,60)
#         self.mqtt_client.publish(MQTT_PUB_CAMERA_TOPIC)
#         self.mqtt_client.loop_start()

#         # ROS setup
#         self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC,Image,self.callback)
        
#         rospy.init_node('mqtt_image_publisher',anonymous=True)

#     def callback(self,data):
#         try:
#             rospy.loginfo("reading from camera feed")
#             cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
#             _,jpeg = cv2.imencode('.jpg',cv_image)
#             self.mqtt_client.publish(MQTT_PUB_CAMERA_TOPIC,jpeg.tobytes())
#             rospy.loginfo("img published to broker at topic %s",MQTT_PUB_CAMERA_TOPIC)
#         except self.bridge.CvBridgeError as e:
#             rospy.logerr(e)


class VehicleNode:
    def __init__(self):
        self.bridge = CvBridge()

        #setup MQTT Client
        self.mqtt_pub_client = mqtt.Client() # To send vehicle data to the cloud node
        self.mqtt_sub_client = mqtt.Client() # To receive segmented images

        self.mqtt_pub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,60)
        self.mqtt_sub_client.connect(MQTT_BROKER_IP,MQTT_BROKER_PORT,61)
        self.mqtt_sub_client.on_message = self.on_mqtt_message #Does the callback job for the subscription of the segmented images

        # List to vehicle's camera (ROS SUB)
        self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC,Image,self.callback)
        #Callback has the MQTT sending to cloud part

        # Receive data from cloud through MQTT topic
        self.mqtt_sub_client.subscribe(MQTT_SUB_SEGMENTED_TOPIC)
        self.ros_pub = rospy.Publisher(ROS_PUB_SEGMENTED_TOPIC, Image, queue_size=10)
        
        self.mqtt_pub_client.loop_start()
        self.mqtt_sub_client.loop_start()

    def callback(self,data):
        try:
            rospy.loginfo_once("reading from camera feed")
            #cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
            cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')

            #cv2.imshow('vehicle',cv_image)
            #cv2.waitKey(0)
            #plt.show(0)
            __,jpeg = cv2.imencode('.jpg',cv_image)
            self.mqtt_pub_client.publish(MQTT_PUB_CAMERA_TOPIC,jpeg.tobytes())
            rospy.loginfo_once("img published to broker at topic %s",MQTT_PUB_CAMERA_TOPIC)
        
        except self.bridge.CvBridgeError as e:
            rospy.logerr(e)
    
    def on_mqtt_message(self, client, userdata, msg):
            rospy.loginfo("Received segmented image from MQTT")
            np_arr = np.frombuffer(msg.payload, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Seems to work with Jpg format
            
            # Display image using matplotlib
            #plt.imshow(cv_image)
            #plt.show(1)

            # Publish image to ROS topic
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.ros_pub.publish(ros_image)

    def run(self):
        rospy.loginfo("starting vehicle node")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':

    #image_subscriber = ImageSubscriber()
    #image_publisher=ImagePublisher()
    rospy.init_node('vehicle_node',anonymous=True)
    vehicle_node = VehicleNode()
    try:
        vehicle_node.run()
    except rospy.ROSInterruptException:
        pass


