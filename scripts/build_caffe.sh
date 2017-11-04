FRAMEWORK_VERSION=1.0

git clone --single-branch --branch $FRAMEWORK_VERSION --recursive https://github.com/BVLC/caffe.git caffe

DIST_DIR=$HOME/frameworks/caffe
mkdir -p $DIST_DIR

cd caffe && \
	mkdir -p build && cd build && \
	cmake .. \
		-DCMAKE_INSTALL_PREFIX=$DIST_DIR \
		-DUSE_ROCKSDB=OFF \
      		-DBUILD_python=OFF \
      		-DUSE_OPENCV=OFF \
      		-DBLAS=open \
      		-DUSE_CUDNN=1 \
      		-DUSE_NCCL=1 \
		-DNCCL_INCLUDE_DIR=/opt/DL/nccl/include \
		-DNCCL_LIBRARIES=/opt/DL/nccl/lib/libnccl.so \
	&& make -j"$(nproc)" install

