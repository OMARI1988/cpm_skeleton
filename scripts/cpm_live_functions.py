# date    06-03-2017
# author  Muhannad Alomari
# email   scmara@leeds.ac.uk
# version 1.0

import rospy
import cv2
import numpy as np
import scipy
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import rospkg
import os
import glob
import getpass
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from cpm_skeleton.msg import cpm_pointer, cpmAction, cpmActionResult
import sys
import actionlib
import shutil

class skeleton_cpm():
    """docstring for cpm"""
    def __init__(self,cam,topic,pub,sav):
        
        # read camera calib
        self.camera_calib = util.read_yaml_calib(cam)

        # save cpm images
        self.save_cpm_img = sav
        if self.save_cpm_img:
            rospy.loginfo("save cpm images.") 

        # get camera topic
        self.image_topic = rospy.resolve_name(topic)

        # initialize published
        self.pub = pub
        if self.pub:
            rospy.loginfo("publish cpm images")
            self.image_pub = rospy.Publisher("/cpm_skeleton_image", Image, queue_size=1)
        else:
            rospy.loginfo("don't publish cpm images")
        
        # cpm init stuff
        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        self.cpm_path = self.rospack.get_path('cpm_skeleton')
        self.conf = 1            	# using config file 1, for the full body detector
        self.param, self.model = config_reader(self.conf)
        self.boxsize = self.model['boxsize']
        self.npart = self.model['np']
        self.limbs_names = ['head','neck', 'right_shoulder', 'right_elbow', 'right_hand', 'left_shoulder', 'left_elbow', 'left_hand',
            'right_hip', 'right_knee', 'right_foot', 'left_hip', 'left_knee', 'left_foot']
        self.colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
        [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
        self.stickwidth = 6
        self.dist_threshold = 1.5   	# less than 1.5 meters ignore the skeleton
        self.depth_thresh = .35     	# any more different in depth than this with openni, use openni
        self.finished_processing = 0   	# a flag to indicate that we finished processing allavailable  data
        self.threshold = 10		# remove any folder <= 10 detections
        self._initiliase_cpm()
        self.processing = 0
        self.image_ready = 0

        # subscribe to camera topic
        rospy.Subscriber(self.image_topic, Image, self.detect_and_draw)

    def _initiliase_cpm(self):
        sys.stdout = open(os.devnull,"w")
        if self.param['use_gpu']:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        caffe.set_device(self.param['GPUdeviceNumber']) # set to your device!
        self.person_net = caffe.Net(self.model['deployFile_person'], self.model['caffemodel_person'], caffe.TEST)
        self.pose_net = caffe.Net(self.model['deployFile'], self.model['caffemodel'], caffe.TEST)
        sys.stdout = sys.__stdout__

    def detect_and_draw(self,imgmsg):
        if not self.processing:
            img = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding="passthrough")
            img = img[:,:,0:3]
            #cv2.imshow('camera feed',img)
            #cv2.waitKey(1)
            self.image = img
            self.image_ready = 1
        
        #self._process_images(img)#, depth, skl)

    def _process_images(self,img):#,depth):
        self.processing = 1
        self.image_ready = 0

        # main loop
        start = time.time()

        # block 1
        scale = self.boxsize/(img.shape[0] * 1.0)
        imageToTest = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest)

        # block 2
        self.person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        self.person_net.reshape()

        # block 3
        self.person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        output_blobs = self.person_net.forward()
        person_map = np.squeeze(self.person_net.blobs[output_blobs.keys()[0]].data)

        # block 4
        person_map_resized = cv2.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
        maxima = (person_map_resized == data_max)
        diff = (data_max > 0.5)
        maxima[diff == 0] = 0
        x = np.nonzero(maxima)[1]
        y = np.nonzero(maxima)[0]

        # block 5
        num_people = x.size
        person_image = np.ones((self.model['boxsize'], self.model['boxsize'], 3, num_people)) * 128
        for p in range(num_people):
	        x_max = x[p]+self.model['boxsize']/2
	        x_min = x[p]-self.model['boxsize']/2
	        y_max = y[p]+self.model['boxsize']/2
	        y_min = y[p]-self.model['boxsize']/2
	        if x_min < 0:
		        xn_min = x_min*-1
		        x_min = 0
	        else:
		        xn_min = 0
	        if x_max > imageToTest.shape[1]:
		        xn_max = self.model['boxsize'] - (x_max-imageToTest.shape[1])
		        x_max = imageToTest.shape[1]
	        else:
		        xn_max = self.model['boxsize']
	        if y_min < 0:
		        yn_min = y_min*-1
		        y_min = 0
	        else:
		        yn_min = 0
	        if y_max > imageToTest.shape[0]:
		        yn_max = self.model['boxsize'] - (y_max-imageToTest.shape[0])
		        y_max = imageToTest.shape[0]
	        else:
		        yn_max = self.model['boxsize']
	        person_image[yn_min:yn_max, xn_min:xn_max, :, p] = imageToTest[y_min:y_max, x_min:x_max, :]

        # block 6
        gaussian_map = np.zeros((self.model['boxsize'], self.model['boxsize']))
        x_p = np.arange(self.model['boxsize'])
        y_p = np.arange(self.model['boxsize'])
        part1 = [(x_p - self.model['boxsize']/2) * (x_p - self.model['boxsize']/2), np.ones(self.model['boxsize'])]
        part2 = [np.ones(self.model['boxsize']), (y_p - self.model['boxsize']/2) * (y_p - self.model['boxsize']/2)]
        dist_sq = np.transpose(np.matrix(part1))*np.matrix(part2)
        exponent = dist_sq / 2.0 / self.model['sigma'] / self.model['sigma']
        gaussian_map = np.exp(-exponent)

        # block 7
        output_blobs_array = [dict() for dummy in range(num_people)]
        for p in range(num_people):
            input_4ch = np.ones((self.model['boxsize'], self.model['boxsize'], 4))
            input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 # normalize to [-0.5, 0.5]
            input_4ch[:,:,3] = gaussian_map
            self.pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
            if self.conf == 4:
	         output_blobs_array[p] = copy.deepcopy(self.pose_net.forward()['Mconv5_stage4'])
            else:
	         output_blobs_array[p] = copy.deepcopy(self.pose_net.forward()['Mconv7_stage6'])

        # block 8
        for p in range(num_people):
            print('Person %d' % p)
            for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
	        part_map = output_blobs_array[p][0,part,:,:]
	        part_map_resized = cv2.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC) #only for displaying

        # block 9
        prediction = np.zeros((14, 2, num_people))
        for p in range(num_people):
            for part in range(14):
	        part_map = output_blobs_array[p][0, part, :, :]
	        part_map_resized = cv2.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
	        prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
            # mapped back on full image
            prediction[:,0,p] = prediction[:,0,p] - (self.model['boxsize']/2) + y[p]
            prediction[:,1,p] = prediction[:,1,p] - (self.model['boxsize']/2) + x[p]

        # block 10
        limbs = self.model['limbs']
        stickwidth = 6
        colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
        [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
        canvas = imageToTest.copy()
        if num_people>0:
            canvas = np.multiply(canvas,0.2,casting="unsafe")
            #canvas *= .6 # for transparency
            cur_canvas = imageToTest.copy() #np.zeros(canvas.shape,dtype=np.uint8)
            for p in range(num_people):
                for part in range(self.model['np']):
	            cv2.circle(canvas, (int(prediction[part, 1, p]), int(prediction[part, 0, p])), 3, (0, 0, 0), -1)
                for l in range(limbs.shape[0]):
	            X = prediction[limbs[l,:]-1, 0, p]
	            Y = prediction[limbs[l,:]-1, 1, p]
	            mX = np.mean(X)
	            mY = np.mean(Y)
	            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
	            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
	            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
	            cv2.fillConvexPoly(cur_canvas, polygon, colors[l])
                    #print int(mY),int(mX),int(length/2)
            canvas = np.add(canvas,np.multiply(cur_canvas,0.8,casting="unsafe"),casting="unsafe") # for transparency

        canvas = canvas.astype(np.uint8)
	#print canvas
        name = ''
        print 'image processed in: %1.3f sec' % (time.time()-start)
        #util.showBGRimage(name+'_results',canvas,1)
        if self.pub:
            sys.stdout = open(os.devnull, "w")
            msg = self.bridge.cv2_to_imgmsg(canvas, "rgb8")
            sys.stdout = sys.__stdout__
            self.image_pub.publish(msg)
        self.processing = 0

