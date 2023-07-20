import rospy
from sensor_msgs.msg import Image
import paho.mqtt.client as mqtt
from PIL import Image as PilImage
import numpy as np

MQTT_BROKER_IP = "localhost"
MQTT_BROKER_PORT = 1883

# ROS vehicle camera image publisher
ROS_SUB_CAMERA_TOPIC = "/vehicle_ros_camera"
# MQTT publisher to broker
MQTT_PUB_CAMERA_TOPIC = "/vehicle_camera"

# MQTT segmented image subscriber
MQTT_SUB_SEGMENTED_TOPIC = "/segmented_images"
# ROS topic for publishing segmented image to vehicle's ROS network
ROS_PUB_SEGMENTED_TOPIC = "/segmented_ros_images"

ROS_RATE = 10 #10hz

class VehicleNode:

    def __init__(self):

        # setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(MQTT_BROKER_IP, MQTT_BROKER_PORT, 60)

        # listens to vehicle ros camera
        self.ros_sub = rospy.Subscriber(ROS_SUB_CAMERA_TOPIC, Image, self.callback)

        # listens to segmented images mqtt topic
        self.mqtt_client.subscribe(MQTT_SUB_SEGMENTED_TOPIC, self.segmented_callback)
        # ros publisher to vehicle ros network topics
        self.ros_pub = rospy.Publisher(ROS_PUB_SEGMENTED_TOPIC, Image, queue_size=10)

        self.mqtt_client.loop_start()

    # vehicle camera subscriber callback
    # applies img processing
    # publishes to mqtt broker 
    def callback(self, data):
        try:
            rospy.loginfo("heard camera image for vehicle ROS")

            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            ret, jpeg = cv2.imencode('.jpg', cv_image)
            rospy.loginfo("vehicle img processed")

            self.mqtt_client.publish(MQTT_PUB_CAMERA_TOPIC, jpeg.tobytes())
            # self.mqtt_client.publish(self.topic, image)
            rospy.loginfo("img published to broker at topic %s", MQTT_PUB_CAMERA_TOPIC)

            print('publishing to ros: ', cv_image)
        except Error as e:
            print(e)

    # callback on receiving the segmented image from mqtt broker
    def segmented_callback(self, data):
        try:
            rospy.loginfo("raw segmented img received at vehicle node")

            # process the recieved image
            np_arr = np.fromstring(message.payload, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            # print(ros_image)
            
            rospy.loginfo("raw segmented img processed at vehicle node")

            # publish to ros
            self.ros_pub.publish(ros_image)
            rospy.loginfo("processed segmented img published to ROS at vehicle node")

        except CvBridgeError as e:
            print(e)  

    def run(self):
        rospy.loginfo("starting vehicle node")
        rate = rospy.Rate(ROS_RATE)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('vehicle_node', anonymous=True)
    vehicle_node = VehicleNode()
    try:
        vehicle_node.run()
    except rospy.ROSInterruptException:
        pass
