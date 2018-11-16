package caffe

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -DBLAS=open -Wno-unused-local-typedef
// #cgo darwin CXXFLAGS: -I/opt/caffe/include -I/usr/local/opt/openblas/include -DCPU_ONLY=1
// #cgo darwin LDFLAGS: -lcaffe -L/opt/caffe/lib
// #cgo !darwin,nogpu CXXFLGAS: -DCPU_ONLY=1
// #cgo !darwin,!nogpu CXXFLAGS: -I/usr/local/cuda/include
// #cgo !darwin,!nogpu LDFLAGS: -lcublas -lcudnn -L/usr/local/cuda/lib64 -L/usr/local/cuda
// #cgo linux,amd64 CXXFLAGS: -I/opt/caffe/include
// #cgo linux,amd64 LDFLAGS: -lcaffe -L/opt/caffe/lib
// #cgo linux,ppc64le,powerai CXXFLAGS: -I/opt/DL/caffe/include
// #cgo linux,ppc64le,powerai LDFLAGS: -L/opt/DL/caffe/lib
// #cgo linux,ppc64le,!powerai CXXFLAGS: -I/opt/caffe/include
// #cgo linux,ppc64le,!powerai LDFLAGS: -lcaffe -L/opt/caffe/lib
import "C"
