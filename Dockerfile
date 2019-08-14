# docker build -t ubuntu1604py36
FROM ubuntu:16.04

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    apt-get install -y software-properties-common vim
    
RUN apt-get update -y

RUN apt-get install -y build-essential python python-pip && \
    apt-get install -y git apt-utils

# ==================================================================
# update pip
# ------------------------------------------------------------------ 
RUN python -m pip install pip --upgrade 
    
# ==================================================================
# python
# ------------------------------------------------------------------    
RUN python -m pip install numpy==1.13.3 \
                          scipy==0.19.1 \
                          opencv-python \
                          matplotlib \
                          scikit-image==0.13.0 \
                          scikit-learn==0.19.1 \
                          imageio \
                          tensorboardX \
                          pyyaml \
                          IPython \
                          h5py==2.7.0 \
                          Keras==2.1.2 \
                          tensorflow==1.4.1
                             
# ==================================================================
# tools
# ------------------------------------------------------------------    
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y python-tk

# ==================================================================
# ROS
# ------------------------------------------------------------------ 
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y --allow-unauthenticated ros-kinetic-desktop && \
    apt-get install -y --allow-unauthenticated python-rosinstall python-rosdep python-vcstools python-catkin-tools && \
    apt-get install -y --allow-unauthenticated ros-kinetic-tf2-geometry-msgs
    
# bootstrap rosdep
RUN rosdep init
RUN rosdep update

RUN pip install rospkg
    
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* 
