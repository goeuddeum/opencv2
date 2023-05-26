#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
def imgCallback(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    cv2.imshow("my_image", frame)
    cv2.waitKey(3)


rospy.init_node("my_cam_sub")
sub = rospy.Subscriber("my_image", Image, imgCallback, queue_size=10)
rospy.spin() 
cv2.destroyAllWindows()

