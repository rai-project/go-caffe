#!/bin/sh

FRAMEWORK_VERSION=master
CAFFE_SRC_DIR=$HOME/code/caffe
CAFFE_DIST_DIR=/opt/caffe

if [ ! -d "$CAFFE_SRC_DIR" ]; then
  git clone --single-branch --branch $FRAMEWORK_VERSION --recursive https://github.com/BVLC/caffe.git $CAFFE_SRC_DIR
fi

if [ ! -d "$CAFFE_DIST_DIR" ]; then
  mkdir -p $CAFFE_DIST_DIR
fi

cd $CAFFE_SRC_DIR && rm -rf build && mkdir build && cd build && \
	cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_DIST_DIR \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_CXX_FLAGS=-std=c++11 \
    -DBLAS=open \
    -DBUILD_python=ON \
    -DUSE_OPENCV=OFF \
    -DCPU_ONLY=ON \
    -DUSE_NCCL=OFF \
    && make -j"$(nproc)" install
