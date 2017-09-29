package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system
// #cgo CXXFLAGS: -DBLAS=open -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo darwin CXXFLAGS: -I/opt/caffe/include -I/usr/local/opt/openblas/include
// #cgo darwin LDFLAGS: -L/opt/caffe/lib  -L/usr/local/opt/openblas/lib
// #cgo ppc64le CXXFLAGS: -I/home/carml/frameworks/caffe/include
// #cgo ppc64le LDFLAGS: -L/home/carml/frameworks/caffe/lib
import "C"
