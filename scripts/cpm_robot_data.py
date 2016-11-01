import rospy
import cpm_functions

sk_cpm = cpm_functions.skeleton_cpm()
count = 1
for rgb,depth,skl in zip(sk_cpm.rgb_files,sk_cpm.dpt_files,sk_cpm.skl_files):
    if count >= sk_cpm.frame_num:
        sk_cpm._process_images(rgb,depth,skl)
    count+=1

