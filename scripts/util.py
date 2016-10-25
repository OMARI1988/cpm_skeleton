import numpy as np
from cStringIO import StringIO
import PIL.Image
from IPython.display import Image, display
import cv2
import getpass

def make_X(x,z):
    return int(x*525.0/z +319.5)

def make_Y(y,z):
   return int(y*(-525.0)/z+ 239.5)

def scale_and_centre_torso(f1,img,sz):
    margin=0.3 #(percentage of the skeleton length) padding around the skeleton to prevent features being cropped out
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
    max_x = make_X(max_x,skeleton_data[max_x_id][2])
    min_x = make_X(min_x,skeleton_data[min_x_id][2])
    max_y = rows-make_Y(max_y,skeleton_data[max_y_id][2])
    min_y = rows-make_Y(min_y,skeleton_data[min_y_id][2])
    torso_x = make_X(skeleton_data["torso"][0],skeleton_data["torso"][2]);
    torso_y = rows-make_Y(skeleton_data["torso"][1],skeleton_data["torso"][2]);
    scaling_f=max(max_y-min_y,max_x-min_x)
    scaling_f_2=int(max(torso_y-min_y,torso_x-min_x)+(margin*scaling_f))
    M = np.float32([[1,0,-(torso_x-scaling_f_2)],[0,1,-(torso_y-scaling_f_2)]])
    dst = cv2.warpAffine(img,M,(cols,rows),borderValue=0.8)
    crop_dst = cv2.resize(dst[0:2*scaling_f_2,0:2*scaling_f_2], (sz,sz))
    return crop_dst


def showBGRimage(name, a, fmt='jpg'):
    a = np.uint8(np.clip(a, 0, 255))
    cv2.imshow(name,a)
    cv2.waitKey(2000)
    cv2.imwrite(name+'.'+fmt,a)

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

