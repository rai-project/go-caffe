package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system -L/usr/local/lib
// #cgo CXXFLAGS: -std=c++14 -I/usr/local/include/ -I${SRCDIR}/cbits -O3 -Wall
// #cgo darwin CXXFLAGS: -DCPU_ONLY=1 -DBLAS=open -I/usr/local/opt/openblas/include
// #cgo darwin LDFLAGS: -L/usr/local/opt/openblas/lib
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

type Predictor struct {
	ctx C.PredictorContext
}

func New(modelFile, trainFile string) (*Predictor, error) {
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	if !com.IsFile(trainFile) {
		return nil, errors.Errorf("file %s not found", trainFile)
	}
	return &Predictor{
		ctx: C.New(C.CString(modelFile), C.CString(trainFile)),
	}, nil
}

func (p *Predictor) Predict(imageData []float32) ([]Prediction, error) {
	ptr := (*C.float)(unsafe.Pointer(&imageData[0]))
	r := C.Predict(p.ctx, ptr)
	js := C.GoString(r)
	return nil, nil
}

func (p *Predictor) Close() {
	C.Delete(p.ctx)
}

func SetUseCPU() {
	e := SolverParameter_SolverMode_value["CPU"]
	C.SetMode(C.int(e))
}

func SetUseGPU() {
	e := SolverParameter_SolverMode_value["GPU"]
	C.SetMode(C.int(e))
}
