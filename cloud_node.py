import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import rosbag
import time
import paho.mqtt.client as mqtt
import frozen_graph_runner

MQTT_BROKER_IP = "localhost"
MQTT_SUB_CAMERA_TOPIC = "/vehicle_camera"
ROS_PUB_CAMERA_TOPIC = "/cloud_ros_camera"
MQTT_PUB_SEGMENTED_TOPIC = "/segmented_images"

class CloudNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(MQTT_BROKER_IP, 1883, 60)
        self.mqtt_client.subscribe(MQTT_SUB_CAMERA_TOPIC, qos=2)

        self.mqtt_callback()
        self.mqtt_client.loop_start()

    def mqtt_callback(self):
        try:
            rospy.loginfo("Heard camera image at cloud node from MQTT broker")

            # Load the rosbag
            bag = rosbag.Bag('left_camera_templergraben.bag')
            bridge = CvBridge()

            for msg in bag.read_messages(topics=[ROS_PUB_CAMERA_TOPIC]):
                input_image = bridge.imgmsg_to_cv2(msg, "bgr8")

                rospy.loginfo("Image processed; starting inference")
                segmented_image = self.perform_inference(input_image)
                rospy.loginfo("Inference completed")

                ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                self.mqtt_client.publish(MQTT_PUB_SEGMENTED_TOPIC, jpeg.tobytes())
                rospy.sleep(0.1)  # Delay between image updates

            bag.close() 

        except CvBridgeError as e:
            print(e)

    def perform_inference(self, input_image):
        prediction = frozen_graph_runner.inference(input_image)
        return prediction

if __name__ == '__main__':
    rospy.init_node('cloud_node', anonymous=True)
    rospy.loginfo("Initialized and starting cloud inference node")
    cloud_node = CloudNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
