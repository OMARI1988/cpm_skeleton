import rospy
import numpy as np
from cStringIO import StringIO
import PIL.Image
from IPython.display import Image, display
import cv2
import getpass
import sys
import yaml

def make_X(x,z,fx,fy,cx,cy):
    return int(x*fx/z +cx)

def make_Y(y,z,fx,fy,cx,cy):
    return int(y*(-fy)/z+ cy)

def read_yaml_calib(dir1):
    try:
        f= open(dir1,'r')
        X=yaml.load(f)
        fx=X["camera_matrix"]["data"][0]
        cx=X["camera_matrix"]["data"][2]
        fy=X["camera_matrix"]["data"][4]
        cy=X["camera_matrix"]["data"][5]
        rospy.loginfo("reading camera calibration completed.")
    except:
        rospy.loginfo("Could not read camera calibration file, using default parameters.")
        fx=525.0
        fy=525.0
        cx=319.5
        cy=239.5
    return [fx,fy,cx,cy]

def get_openni_values(f1):
    openni_values = {}
    time = 'time:'
    for count,line in enumerate(f1):
        if ':' in line: 
            time=line
            continue
        line = line.split(',')
        openni_values[line[0]] = {}
        x2d = float(line[1])
        y2d = float(line[2])
        x = float(line[3])
        y = float(line[4])
        z = float(line[5])
        c = float(line[6])
        openni_values[line[0]]['x2d']=x2d
        openni_values[line[0]]['y2d']=y2d
        openni_values[line[0]]['x']=x
        openni_values[line[0]]['y']=y
        openni_values[line[0]]['z']=z
    return openni_values,time


def get_correct_person(openni_values,scale,camera_calib,X,Y):
    [fx,fy,cx,cy] = camera_calib
    x = openni_values['torso']['x']
    y = openni_values['torso']['y']
    z = openni_values['torso']['z']
    #2D data
    x2d = int(int(x*fx/z*1 +cx)*scale);
    y2d = int(int(y*fy/z*1+cy)*scale);

    distance_min = 1000
    distance_threshold = 50
    person = []
    id = 0
    for x1,y1 in zip(X,Y):
        distance = np.sqrt((x1-x2d)**2 + (y1-y2d)**2)
        if distance < distance_min and distance < distance_threshold:
            person = id
        id+=1
    if person != []:
        x = [X[person]]; y = [Y[person]]
    else:
        x = []; y = []
    return [x,y]


def scale_and_centre_torso(f1,img,sz):
    [fx,fy,cx,cy] = read_yaml_calib()
    margin=0.5 #(percentage of the skeleton length) padding around the skeleton to prevent features being cropped out
    cols=img.shape[1]
    rows=img.shape[0]
    max_x, max_y=-1000,-1000
    min_x, min_y= 1000, 1000
    min_x_id,min_y_id,max_x_id,max_y_id=-1,-1,-1,-1
    skeleton_data=dict()
    #read the skeleton data from skl file
    for count,line in enumerate(f1):
        if count == 0:
            t = line.split(':')[1].split('\n')[0]
        # read the joint name
        elif (count-1)%11 == 0:
            j = line.split('\n')[0]
            skeleton_data[j] = []
        # read the x value
        elif (count-1)%11 == 2:
            x = float(line.split('\n')[0].split(':')[1])
            skeleton_data[j].append(x)
            if x>max_x:
                max_x=x
                max_x_id=j
            if x<min_x:
                min_x=x
                min_x_id=j
        # read the y value
        elif (count-1)%11 == 3:
            y = float(line.split('\n')[0].split(':')[1])
            skeleton_data[j].append(y)
            if y>max_y:
                max_y=y
                max_y_id=j
            if y<min_y:
                min_y=y
                min_y_id=j
        # read the z value
        elif (count-1)%11 == 4:
            z = float(line.split('\n')[0].split(':')[1])
            skeleton_data[j].append(z)
    #scale the image to center around the torso
    max_x = make_X(max_x,skeleton_data[max_x_id][2],fx,fy,cx,cy)
    min_x = make_X(min_x,skeleton_data[min_x_id][2],fx,fy,cx,cy)
    max_y = rows-make_Y(max_y,skeleton_data[max_y_id][2],fx,fy,cx,cy)
    min_y = rows-make_Y(min_y,skeleton_data[min_y_id][2],fx,fy,cx,cy)
    torso_x = make_X(skeleton_data["torso"][0],skeleton_data["torso"][2],fx,fy,cx,cy);
    torso_y = rows-make_Y(skeleton_data["torso"][1],skeleton_data["torso"][2],fx,fy,cx,cy);
    scaling_f=max(max_y-min_y,max_x-min_x)
    scaling_f_2=int(max(torso_y-min_y,torso_x-min_x)+(margin*scaling_f))
    M = np.float32([[1,0,-(torso_x-scaling_f_2)],[0,1,-(torso_y-scaling_f_2)]])
    dst = cv2.warpAffine(img,M,(cols,rows),borderValue=0.8)
    crop_dst = cv2.resize(dst[0:2*scaling_f_2,0:2*scaling_f_2], (sz,sz))
    return crop_dst

def saveBGRimage(name, a, fmt='jpg'):
    a = np.uint8(np.clip(a, 0, 255))
    #cv2.imshow(name,a)
    #cv2.waitKey(2000)
    cv2.imwrite(name+'.'+fmt,a)

def showBGRimage(name, a, t, fmt='jpg'):
    a = np.uint8(np.clip(a, 0, 255))
    cv2.imshow(name,a)
    cv2.waitKey(t)
    #cv2.imwrite(name+'.'+fmt,a)

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%8==0) else 8 - (h % 8) # down
    pad[3] = 0 if (w*8==0) else 8 - (w % 8) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + 128, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]* + 128, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]* + 128, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + 128, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

