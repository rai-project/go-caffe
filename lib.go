package caffe

// #cgo LDFLAGS: -lstdc++ -lglog -lboost_system -lcaffe
// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -Wall -g
// #cgo debug CXXFLAGS: -O0 -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -DBLAS=open
// #cgo darwin CXXFLAGS: -I/opt/caffe/include -I/usr/local/opt/openblas/include -DCPU_ONLY=1
// #cgo linux LDFLAGS: -lcublas -lcudnn -L/usr/local/cuda/lib64 -L/usr/local/cuda
// #cgo linux CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64 CXXFLAGS: -I/opt/caffe/include -I/opt/caffe/include
// #cgo linux,amd64 LDFLAGS: -L/opt/caffe/lib
// #cgo ppc64le,powerai1.5.3-bvlc CXXFLAGS: -I/opt/DL/caffe-bvlc/include -I/opt/DL/protobuf/include
// #cgo ppc64le,powerai1.5.3-bvlc LDFLAGS: -L/opt/DL/caffe-bvlc/lib  -L/opt/DL/protobuf/lib
// #cgo ppc64le,powerai1.5.3-ibm CXXFLAGS: -I/opt/DL/caffe-ibm/include -I/opt/DL/protobuf/include
// #cgo ppc64le,powerai1.5.3-ibm LDFLAGS: -L/opt/DL/caffe-ibm/lib -L/opt/DL/protobuf/lib
// #cgo ppc64le,powerai CXXFLAGS: -I/opt/DL/caffe/include
// #cgo ppc64le,powerai LDFLAGS: -L/opt/DL/caffe/lib
// #cgo ppc64le,!powerai,!powerai1.5.3-bvlc,!powerai1.5.3-ibm CXXFLAGS: -I/opt/caffe/include
// #cgo ppc64le,!powerai,!powerai1.5.3-bvlc,!powerai1.5.3-ibm LDFLAGS: -L/opt/caffe/lib
import "C"
