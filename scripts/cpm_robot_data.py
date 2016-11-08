#!/usr/bin/env python

import rospy
import cpm_functions
import sys

# to run on bob for testing
# rosrun actionlib axclient.py /cpm_action

if __name__== "__main__":

    rospy.init_node('cpm_skeleton', anonymous=True)
    if len(sys.argv) < 3:
        print("usage: cpm_robot_data.py takes three arguments")
    else:
        rem = rospy.get_param("~remove_rgb","")
        cam = rospy.get_param("~camera_calibration","")
        pub = rospy.get_param("~publish_images","")
        sk_cpm = cpm_functions.skeleton_cpm(cam,rem,pub)
        rospy.spin()
