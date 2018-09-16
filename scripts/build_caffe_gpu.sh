FRAMEWORK_VERSION=1.0
CAFFE_SRC_DIR=$HOME/code/caffe

git clone --single-branch --branch $FRAMEWORK_VERSION --recursive https://github.com/BVLC/caffe.git $CAFFE_SRC_DIR

CAFFE_DIST_DIR=/opt/caffe
mkdir -p $CAFFE_DIST_DIR

cd $CAFFE_SRC_DIR && mkdir -p build && cd build && \
	cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CAFFE_DIST_DIR \
    -DCMAKE_CXX_STANDARD=11 \
    -DBLAS=open \
    -DBUILD_python=OFF \
    -DUSE_OPENCV=OFF \
    -DUSE_NCCL=OFF \
    -DCPU_ONLY=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="30 35 50 52 60 61 70" \
    -DCUDA_ARCH_PTX="30 35 50 52 60 61 70" \
    -DUSE_CUDNN=ON \
    -DCUDNN_ROOT=/usr/include \
	&& make -j"$(nproc)" install
