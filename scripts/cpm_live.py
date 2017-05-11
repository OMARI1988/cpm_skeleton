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
        pub = rospy.get_param("~publish_images","True")
        save = rospy.get_param("~save_images","")
        im_topic = rospy.get_param("~image","/head_xtion/rgb/image_raw")		# subscribed to image topic
        dp_topic = rospy.get_param("~depth","/head_xtion/depth_registered/sw_registered/image_rect")	# subscribed to depth topic
        sk_topic = rospy.get_param("~skeleton","/skeleton_data/incremental")		# subscribed to openni skeleton topic
        self.sk_cpm = cpm_live_functions.skeleton_cpm(cam, im_topic, dp_topic, sk_topic, pub, save)
        counter = 0
        r = rospy.Rate(10) # 30hz
        while not rospy.is_shutdown():
            if self.sk_cpm.image_ready and self.sk_cpm.depth_ready and self.sk_cpm.openni_ready:
                counter = 0
                for userID in self.sk_cpm.openni_data:
                    if "img_xy" in self.sk_cpm.openni_data[userID].keys():
                        img    = self.sk_cpm.openni_data[userID]["process_img"]
                        depth  = self.sk_cpm.openni_data[userID]["process_depth"]
                        img_xy = self.sk_cpm.openni_data[userID]["img_xy"]
                        self.sk_cpm._process_images(img, depth, img_xy, userID)
                self.sk_cpm._publish()
            else:
                if self.sk_cpm.image_ready and counter>10:
                    self.sk_cpm._publish()
                counter += 1
                r.sleep()


def main():
    rospy.init_node('cpm_live')
    print "initialising CPM"
    cpm = live_cpm()

if __name__ == '__main__':
    main()
    

