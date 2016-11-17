
# date    07-11-2016
# author  Muhannad Alomari
# email   scmara@leeds.ac.uk
# version 1.0

import rospy
import cv2 as cv
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
from mongodb_store.message_store import MessageStoreProxy
from cpm_skeleton.msg import cpm_pointer, cpmAction, cpmActionResult
import sys
import actionlib
from activity_data.msg import HumanActivities
import shutil

class skeleton_cpm():
    """docstring for cpm"""
    def __init__(self,cam,rem,pub,sav):
        
        # read camera calib
        self.camera_calib = util.read_yaml_calib(cam)

        # remove rgb images
        self.rgb_remove = rem
        if self.rgb_remove:
            rospy.loginfo("remove rgb images from the dataset.") 

        # save cpm images
        self.save_cpm_img = sav
        if self.save_cpm_img:
            rospy.loginfo("save cpm images.") 

        # initialize published
        self.pub = pub
        if self.pub:
            rospy.loginfo("publish cpm images")
            self.image_pub = rospy.Publisher("/cpm_skeleton_image", Image, queue_size=1)
            self.bridge = CvBridge()
        else:
            rospy.loginfo("don't publish cpm images")

        # mongo stuff
        self.msg_store = MessageStoreProxy(database='message_store', collection='cpm_stats')
        self.msg_store_learning = MessageStoreProxy(database='message_store', collection='activity_learning')

        # open dataset folder
        self.directory = '/home/'+getpass.getuser()+'/SkeletonDataset/no_consent/'
        if not os.path.isdir(self.directory):
            rospy.loginfo(self.directory+" does not exist. Please make sure there is a dataset on this pc")
            sys.exit(1)
        self.dates = sorted(os.listdir(self.directory))
        self.get_dates_to_process()
        if not self.read_mongo_success:
              self.folder = 0
              self.userid = 0 
              self._read_files()
        #sys.exit(1)
        
        # cpm init stuff
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
        self.cpm_stats_file = '/home/'+getpass.getuser()+'/SkeletonDataset/cpm_stats.txt'

        # action server
        self._as = actionlib.SimpleActionServer("cpm_action", cpmAction, \
                    execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def _initiliase_cpm(self):
        if self.param['use_gpu']:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        caffe.set_device(self.param['GPUdeviceNumber']) # set to your device!
        self.person_net = caffe.Net(self.model['deployFile_person'], self.model['caffemodel_person'], caffe.TEST)
        self.pose_net = caffe.Net(self.model['deployFile'], self.model['caffemodel'], caffe.TEST)

    def _cpm_stats(self, start, duration, stop_flag_pre, stop_flag_dur):
        f1 = open(self.cpm_stats_file,'a')
        f1.write('date='+start.split(' ')[0])
        f1.write(', start='+start.split(' ')[1])
        f1.write(', end='+time.strftime("%H:%M:%S"))
        f1.write(', duration='+ str(duration))
        f1.write(', processed='+str(self.processed))
        f1.write(', removed='+str(self.removed))
        f1.write(', images='+str(self.img_processed))
        if self.finished_processing:
            f1.write(', stopped=finisehd all data\n')
        elif stop_flag_pre:
            f1.write(', stopped=preempted\n')
        elif stop_flag_dur:
            f1.write(', stopped=duration\n')

    def execute_cb(self, goal):
        self.processed = 0; self.removed = 0; self.img_processed = 0	# stats counter
        self._initiliase_cpm()
        stats_start = time.strftime("%d-%b-%Y %H:%M:%S")
        start = rospy.Time.now()
        end = rospy.Time.now()
        stop_flag_pre = 0; stop_flag_dur = 0				# stop flags preempt and duration
        duration = goal.duration.secs
        while not self.finished_processing and not stop_flag_pre and not stop_flag_dur:
            self.person_found_flag = 0
            for rgb, depth, skl in zip(self.rgb_files, self.dpt_files, self.skl_files):
                if self._as.is_preempt_requested():
                     stop_flag_pre=1; break
                if (end - start).secs > duration:
                     stop_flag_dur=1; break
                self._process_images(rgb, depth, skl)
                end = rospy.Time.now()
                self.img_processed+=1				# counts the number of processed images
            if not stop_flag_pre and not stop_flag_dur:
                self.processed+=1				# stats counter
                if self.person_found_flag > self.threshold:
                    self.update_last_learning()
                    self.update_last_cpm_date()
                    if self.rgb_remove:    self._remove_rgb_images()	# remove rgb images from directory
                    self.next()				# stats counter
                else:
                    rospy.loginfo('nothing interesting was detected, I will delete this folder!')
                    self.delete_last_learning()
                    self.remove_uuid_folder()
                    self.removed+=1
                    self.next()
                         
        # after the action reset everything
        self._cpm_stats(stats_start, duration, stop_flag_pre, stop_flag_dur)
        self._as.set_succeeded(cpmActionResult())

    def _remove_rgb_images(self):
        rospy.loginfo('removing: '+self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/rgb')
        shutil.rmtree(self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/rgb')
        
    def remove_uuid_folder(self):
        #print len(self.files)
        rospy.loginfo('removing: '+self.directory+self.dates[self.folder]+'/'+self.files[self.userid])
        shutil.rmtree(self.directory+self.dates[self.folder]+'/'+self.files[self.userid])
        self.files = sorted(os.listdir(self.directory+self.dates[self.folder]))
        #print len(self.files)
        #sys.exit(1)

    def delete_last_learning(self):
        uuid = self.files[self.userid].split('_')[-1]
        query = {"uuid" : uuid}
        result = self.msg_store_learning.query(type=HumanActivities._type, message_query=query)
        rospy.loginfo("I removed id from mongodb: "+uuid)
        for (ret,meta) in result:
            self.msg_store_learning.delete(message_id=str(meta['_id']))

    def update_last_cpm_date(self):
        msg = cpm_pointer()
        msg.type = "cpm_skeleton"
        msg.date_ran = time.strftime("%Y-%m-%d")
        msg.last_date_used = self.dates[self.folder]
        msg.uuid = self.files[self.userid]
        print "adding %s to cpm stats store" % msg.uuid
        query = {"type" : msg.type}
        self.msg_store.update(message=msg, message_query=query, upsert=True)

    def update_last_learning(self):
        msg = HumanActivities()
        msg.date = self.dates[self.folder]
        msg.uuid = self.files[self.userid].split('_')[-1]
        msg.time = self.files[self.userid].split('_')[-2]
        msg.cpm = True
        print "adding %s to activity learning store" % msg.uuid
        query = {"uuid" : msg.uuid}
        self.msg_store_learning.update(message=msg, message_query=query, upsert=True)

    def get_dates_to_process(self):
        """ Find the sequence of date folders (on disc) which have not been processed into QSRs.
        ret: self.not_processed_dates - List of date folders to use
        """
        self.read_mongo_success = 0
        for (ret, meta) in self.msg_store.query(cpm_pointer._type):
            if ret.type != "cpm_skeleton": continue
            self.read_mongo_success = 1
            self.folder = self.dates.index(ret.last_date_used)
            self.files = sorted(os.listdir(self.directory+self.dates[self.folder]))
            self.userid = self.files.index(ret.uuid)
            rospy.loginfo("cpm progress date: "+ret.last_date_used+","+ret.uuid)
            self.next()
        
    def _read_files(self):
        self.files = sorted(os.listdir(self.directory+self.dates[self.folder]))
        self.rgb_dir = self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/rgb/'
        self.dpt_dir = self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/depth/'
        self.skl_dir = self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/skeleton/'
        self.cpm_dir = self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/cpm_skeleton/'
        self.cpm_img_dir = self.directory+self.dates[self.folder]+'/'+self.files[self.userid]+'/cpm_images/'
        self.rgb_files = sorted(glob.glob(self.rgb_dir+"*.jpg"))
        self.dpt_files = sorted(glob.glob(self.dpt_dir+"*.jpg"))
        self.skl_files = sorted(glob.glob(self.skl_dir+"*.txt"))
        rospy.loginfo('Processing userid: '+self.files[self.userid])
        if not os.path.isdir(self.cpm_dir):
            os.mkdir(self.cpm_dir)

        if not os.path.isdir(self.cpm_img_dir) and self.save_cpm_img:
            os.mkdir(self.cpm_img_dir)
            
    def next(self):
        self.userid+=1
        if self.userid >= len(self.files):
            self.userid = 0
            self.folder += 1
        if self.folder >= len(self.dates):
            rospy.loginfo("cpm finished processing all folders")
            self.finished_processing = 1
        else:
            self._read_files() 

    def _process_images(self,test_image,test_depth,test_skl):
        # block 1
        self.test_skl = test_skl
        self.name = test_image.split('.')[0].split('/')[-1]
        start_time = time.time()
        img = cv.imread(test_image)
        depth = cv.imread(test_depth)
        self.scale = self.boxsize/(img.shape[0] * 1.0)
        imageToTest = cv.resize(img, (0,0), fx=self.scale, fy=self.scale, interpolation=cv.INTER_CUBIC)
        depthToTest = cv.resize(depth, (0,0), fx=self.scale, fy=self.scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
        
        # check distance threshold
        f1 = open(self.test_skl,'r')
        self.openni_values, self.openni_time = util.get_openni_values(f1)
        x = []
        y = []
        if self.openni_values['torso']['z'] >= self.dist_threshold:
            # block 2
            self.person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            self.person_net.reshape()
            #person_net.forward(); # dry run to avoid GPU synchronization later in caffe
            
            # block 3
            self.person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            output_blobs = self.person_net.forward()
            person_map = np.squeeze(self.person_net.blobs[output_blobs.keys()[0]].data)
            
            # block 4
            person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
            data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
            maxima = (person_map_resized == data_max)
            diff = (data_max > 0.5)
            maxima[diff == 0] = 0
            x = np.nonzero(maxima)[1]
            y = np.nonzero(maxima)[0]
            # get the right person from openni
            x,y = util.get_correct_person(self.openni_values,self.scale,self.camera_calib,x,y)
            self.x = x; self.y = y
            
        # block 5
        num_people = len(x)
        person_image = np.ones((self.model['boxsize'], self.model['boxsize'], 3, num_people)) * 128
        for p in range(num_people):
            x_max = x[p]+self.model['boxsize']/2
            x_min = x[p]-self.model['boxsize']/2
            y_max = y[p]+self.model['boxsize']/2
            y_min = y[p]-self.model['boxsize']/2
            if x_min < 0:
                xn_min = x_min*-1; x_min = 0
            else:
                xn_min = 0
            if x_max > imageToTest.shape[1]:
                xn_max = self.model['boxsize'] - (x_max-imageToTest.shape[1])
                x_max = imageToTest.shape[1]
            else:
                xn_max = self.model['boxsize']
            if y_min < 0:
                yn_min = y_min*-1; y_min = 0
            else:
                yn_min = 0
            if y_max > imageToTest.shape[0]:
                yn_max = self.model['boxsize'] - (y_max-imageToTest.shape[0]); y_max = imageToTest.shape[0]
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
        #pose_net.forward() # dry run to avoid GPU synchronization later in caffe
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
            for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
                part_map = output_blobs_array[p][0,part,:,:]
                part_map_resized = cv.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv.INTER_CUBIC) #only for displaying
        
        # block 9
        prediction = np.zeros((14, 2, num_people))
        for p in range(num_people):
            for part in range(14):
                part_map = output_blobs_array[p][0, part, :, :]
                part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
                prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
            # mapped back on full image
            prediction[:,0,p] = prediction[:,0,p] - (self.model['boxsize']/2) + y[p]
            prediction[:,1,p] = prediction[:,1,p] - (self.model['boxsize']/2) + x[p]

        # block 10
        limbs = self.model['limbs']
        canvas = imageToTest.copy()
        #canvas *= .5 # for transparency
        canvas = np.multiply(canvas,0.5,casting="unsafe")
        if num_people:
            self.person_found_flag += 1		# this is used to prevent the deletion of the entire folder if noe person is found
            self._get_depth_data(prediction,depthToTest)
            for p in range(num_people):
                cur_canvas = np.zeros(canvas.shape,dtype=np.uint8)
                for l in range(limbs.shape[0]):
                    X = prediction[limbs[l,:]-1, 0, p]
                    Y = prediction[limbs[l,:]-1, 1, p]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), self.stickwidth), int(angle), 0, 360, 1)
                    cv.fillConvexPoly(cur_canvas, polygon, self.colors[l])
                    cv.fillConvexPoly(depthToTest, polygon, self.colors[l])
                cv.circle(cur_canvas,(int(self.x[0]),int(self.y[0])), 3, (250,0,210), -1)
                #canvas += cur_canvas * 0.5 # for transparency
                canvas = np.add(canvas,np.multiply(cur_canvas,0.5,casting="unsafe"),casting="unsafe").astype(np.uint8) # for transparency
            print 'image '+ self.name +' processed in: %2.3f' %(time.time() - start_time), "person found"
        else:
            print 'image '+ self.name +' processed in: %2.3f' %(time.time() - start_time), "person not found"
        vis = np.concatenate((canvas, depthToTest), axis=1)
        # saving cpm images
        if self.save_cpm_img:
            cv.imwrite(self.cpm_img_dir+self.name+'.jpg',vis)
            
        # publishing cpm images
        if self.pub:
            sys.stdout = open(os.devnull, "w")
            msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            sys.stdout = sys.__stdout__
            self.image_pub.publish(msg)
            #cv.imwrite('/home/strands/SkeletonDataset/cpm_images/cpm_'+self.name+'.jpg',vis)
            #### Create CompressedIamge ####
            #msg = CompressedImage()    
            #msg.header.stamp = rospy.Time.now()
            #msg.format = "jpeg"
            #msg.data = vis  # np.array(cv.imencode('.jpg', vis)[1]).tostring()
            #### Publish new image
            #self.image_pub.publish(msg)

    def _get_depth_data(self, prediction, depthToTest):
        [fx,fy,cx,cy] = self.camera_calib
        cpm_file = self.cpm_dir + 'cpm_' +self.test_skl.split('/')[-1]
        f1 = open(cpm_file,'w')
        f1.write(self.openni_time)
        for part,jname in enumerate(self.limbs_names):
            x2d = np.min([int(prediction[part, 0, 0]),367])
            y2d = np.min([int(prediction[part, 1, 0]),490])
            depth_val = depthToTest[x2d, y2d, 0]
            z = (.4)/(20.0)*(depth_val-60.0) + 2.7
            if np.abs(z-self.openni_values[jname]['z'])>self.depth_thresh:
                z = self.openni_values[jname]['z']
            x = (y2d/self.scale-cx)*z/fx
            y = (x2d/self.scale-cy)*z/fy
            f1.write(jname+','+str(x2d)+','+str(y2d)+','+str(x)+','+str(y)+','+str(z)+'\n')
        # add the torso position
        x2d = np.min([int(self.y[0]),367])
        y2d = np.min([int(self.x[0]),490])     
        depth_val = depthToTest[x2d, y2d, 0]
        z = (.4)/(20.0)*(depth_val-60.0) + 2.7
        if np.abs(z-self.openni_values[jname]['z'])>self.depth_thresh:
            z = self.openni_values[jname]['z']   
        x = (y2d/self.scale-cx)*z/fx
        y = (x2d/self.scale-cy)*z/fy
        f1.write('torso'+','+str(x2d)+','+str(y2d)+','+str(x)+','+str(y)+','+str(z)+'\n')
        f1.close()
