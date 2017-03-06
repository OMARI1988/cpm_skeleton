#!/usr/bin/python
# __author__: Muhannad Alomari
# __email__:  scmara@leeds.ac.uk

import cv2
import rospy
import numpy as np
import cpm_live_functions

class live_cpm():
    def __init__(self):
        cam = rospy.get_param("~camera_calibration","")
        pub = rospy.get_param("~publish_images","")
        sav = rospy.get_param("~save_images","")
        topic = rospy.get_param("~image","/cameras/head_camera/image")
        self.sk_cpm = cpm_live_functions.skeleton_cpm(cam,topic,pub,sav)
        r = rospy.Rate(30) # 10hz
        while not rospy.is_shutdown():
            if self.sk_cpm.image_ready:
                self.sk_cpm._process_images(self.sk_cpm.image)
            else:
                r.sleep()


def main():
    rospy.init_node('cpm_live')
    print "initialising CPM"
    cpm = live_cpm()

if __name__ == '__main__':
    main()
    

