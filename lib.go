package caffe

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lcaffe
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -DBLAS=open
// #cgo darwin CXXFLAGS: -I/opt/caffe/include -I/usr/local/opt/openblas/include -DCPU_ONLY=1
// #cgo darwin LDFLAGS: -L/opt/caffe/lib -L/usr/local/opt/openblas/lib
// #cgo linux CXXFLAGS: -I/usr/local/cuda/include
// #cgo ppc64le,powerai CXXFLAGS: -I/opt/DL/caffe/include
// #cgo ppc64le,powerai LDFLAGS: -L/opt/DL/caffe/lib
// #cgo ppc64le,!powerai CXXFLAGS: -I/opt/caffe/include
// #cgo ppc64le,!powerai LDFLAGS: -L/opt/caffe/lib
import "C"
