package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system
// #cgo CXXFLAGS: -DBLAS=open -std=c++11 -I${SRCDIR}/cbits -O1 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo darwin CXXFLAGS: -I/opt/caffe/include -I/usr/local/opt/openblas/include -DCPU_ONLY=1
// #cgo darwin LDFLAGS: -L/opt/caffe/lib  -L/usr/local/opt/openblas/lib
// #cgo ppc64le,powerai CXXFLAGS: -I/opt/DL/caffe/include -I/usr/local/cuda/include
// #cgo ppc64le,powerai LDFLAGS: -L/opt/DL/caffe/lib
// #cgo ppc64le,!powerai CXXFLAGS: -I/opt/caffe/include -I/home/carml/frameworks/caffe/include -I/usr/local/cuda/include
// #cgo ppc64le,!powerai LDFLAGS: -L/opt/caffe/lib -L/home/carml/frameworks/caffe/lib
// #cgo linux CXXFLAGS: -I/home/as29/my_caffe/caffe/build/install/include -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -L/home/as29/my_caffe/caffe/build/install/lib
import "C"
