#!/usr/bin/env python
# license removed for brevity
import rospy
import cpm_functions
import sys

rospy.init_node('cpm_skeleton', anonymous=True)
sk_cpm = cpm_functions.skeleton_cpm()

rospy.spin()

#while not rospy.is_shutdown() and not sk_cpm.finished_processing:
#    for rgb,depth,skl in zip(sk_cpm.rgb_files,sk_cpm.dpt_files,sk_cpm.skl_files):
        #sk_cpm._process_images(rgb,depth,skl)
#        pass
    #sk_cpm.update_last_learning_date()
#    sk_cpm.next()
    #sys.exit(1)
