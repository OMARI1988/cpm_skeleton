#!/usr/bin/env python
# license removed for brevity
import rospy
import cpm_functions

rospy.init_node('cpm_skeleton', anonymous=True)
sk_cpm = cpm_functions.skeleton_cpm()

while not rospy.is_shutdown() and not sk_cpm.finished_processing:
    frame = 1
    for rgb,depth,skl in zip(sk_cpm.rgb_files,sk_cpm.dpt_files,sk_cpm.skl_files):
        if frame >= sk_cpm.frame_num:
            #sk_cpm._process_images(rgb,depth,skl)
            pass
        frame+=1
    sk_cpm.next()
