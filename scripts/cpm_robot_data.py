#!/usr/bin/env python
# license removed for brevity
import rospy
import cpm_functions
import sys

# to run on bob for testing
# rosrun actionlib axclient.py /cpm_action

rospy.init_node('cpm_skeleton', anonymous=True)
sk_cpm = cpm_functions.skeleton_cpm()
rospy.spin()
