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

cd $CAFFE_SRC_DIR && mkdir -p build && cd build && \
	cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_DIST_DIR \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_CXX_FLAGS=-std=c++11 \
    -DBLAS=open \
    -DBUILD_python=ON \
    -DUSE_OPENCV=OFF \
    -DCPU_ONLY=OFF \
    -DUSE_CUDNN=ON \
    -DUSE_NCCL=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="30 35 37 50 52 53 60 61 62 70 75" \
    -DCUDA_ARCH_PTX="62 70 75" \
    -DCUDNN_ROOT=/usr/include \
    && make -j"$(nproc)" install
