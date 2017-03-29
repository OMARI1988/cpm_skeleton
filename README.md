# cpm_skeleton
convolutional-pose-machines-release in ros

Contact: Muhannad Alomari, scmara@leeds.ac.uk.

![marker](https://raw.githubusercontent.com/OMARI1988/cpm_skeleton/master/data/rgb_00216_results.jpg)

## If you use this code, please cite:
Paul Duckworth, Muhannad Alomari, James Charles, David C. Hogg and Anthony G. Cohn. "[Latent Dirichlet Allocation for Unsupervised Activity Analysis on an Autonomous Mobile Robot](http://eprints.whiterose.ac.uk/109105/)". In Proceedings of the 31st AAAI Conference on Artificial Intelligence, AAAI-17.

# Convolutional Pose Machines
Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh, "[Convolutional Pose Machines](http://arxiv.org/abs/1602.00134)", CVPR 2016.

This project is licensed under the terms of the GPL v2 license. By using the software, you are agreeing to the terms of the [license agreement](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/LICENSE.md).

# Installation process
## Prerequisites
- Ensure you have `libprotobuf-dev` and `protobuf-compiler` installed:
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
- Ensure you also have `libhdf5-serial-dev` installed:
```
apt-get install libhdf5-serial-dev
```
- Ensure you have `libgflags-dev`, `libgoogle-glog-dev` and `liblmdb-dev` installed:
```
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
- Ensure you have `libleveldb-dev` installed:
```
sudo apt-get install libleveldb-dev
```
- Ensure you have `libsnappy-dev` installed:
```
sudo apt-get install libsnappy-dev
```
- Ensure you have `libatlas-base-dev` and `libatlas-dev` installed:
```
sudo apt-get install libatlas-base-dev libatlas-dev
```

## Installing Caffe
- `mkdir ~/sk_cpm`
- `cd ~/sk_cpm`
- `git clone https://github.com/OMARI1988/caffe.git`

## Installing CUDA 8.0
- download cuda 8.0 from https://developer.nvidia.com/cuda-downloads, make sure to choose linux, x86_64, Ubuntu, 14.04, deb(local)
- `cd ~/wherever you installed cuda`
- `sudo dpkg -i cuda-repo-ubuntu1404_8-0-local_8.0.44-1_amd64.deb`
- `sudo apt-get update`
- `sudo apt-get install cuda`
- As part of the CUDA environment, you should add the following in the .bashrc file of your home folder.
```
export CUDA_HOME=/usr/local/cuda-8.0 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
```

## installing cudNN
- you have to get cudNN (to-do, put the google drive link)
- `sudo pip install protobuf`
- `sudo pip install configobj`
- `sudo pip install --upgrade IPython`

## Building caffe
you have to get cudNN and sudo pip install protobuf and sudo pip install configobj and sudo pip install --upgrade IPython

- `cd ~/sk_cmp/caffe`
- `mkdir build`
- `cd build`
- `cmake ..`
- `make all`
- `make install`
- `make runtest`

## Pythonpath for cuda
- add python path of caffe to bashrc
- `export PYTHONPATH=$PYTHONPATH:~/sk_cpm/caffe/python/`

## Installing cpm_skeleton in your catkin_Ws
- `cd ~/wherever_your_catkin_ws/src/`
- `git clone https://github.com/OMARI1988/cpm_skeleton.git`
- ` cd ..`
- `catkin_make`

## download the caffe models
- `roscd cpm_skeleton/model/`
- `./get_model.sh`

## first test
- `roscd cpm_skeleton/scripts/`
- `python first_test.py`
- you should see an output image like this for 1 sec

If you get any errors when you are running the `first_test.py` then try to install the following:
```
sudo pip install scikit-image
```

![marker](https://raw.githubusercontent.com/OMARI1988/cpm_skeleton/master/data/rgb_00216_results.jpg)
