from configobj import ConfigObj
import numpy as np
import rospkg

def config_reader(conf):
    rospack = rospkg.RosPack()
    path = rospack.get_path('cpm_skeleton')
    config = ConfigObj(path+'/config/config_'+str(conf))

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['np'] = int(model['np'])
    num_limb = len(model['limbs'])/2
    model['limbs'] = np.array(model['limbs']).reshape((num_limb, 2))
    model['limbs'] = model['limbs'].astype(np.int)
    model['sigma'] = float(model['sigma'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    #param['octave'] = int(param['octave'])
    param['use_gpu'] = int(param['use_gpu'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])
    #print param.keys()
    #print model.keys()
    model['deployFile_person'] = path+'/model/'+model['deployFile_person']
    model['caffemodel_person'] = path+'/model/'+model['caffemodel_person']
    model['deployFile'] = path+'/model/'+model['deployFile']
    model['caffemodel'] = path+'/model/'+model['caffemodel']
    return param, model

if __name__ == "__main__":
    # print 'test'
    config_reader()
