package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system -L/usr/local/lib  -L/opt/caffe/lib
// #cgo CXXFLAGS: -std=c++11 -I/usr/local/include/ -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1 -I/opt/caffe/include
// #cgo darwin CXXFLAGS: -DBLAS=open -I/usr/local/opt/openblas/include
// #cgo darwin LDFLAGS: -L/usr/local/opt/openblas/lib
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"encoding/json"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

const (
	CPUMode = 0
	GPUMode = 1
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

func (p *Predictor) Predict(imageData []float32) (Predictions, error) {
	ptr := (*C.float)(unsafe.Pointer(&imageData[0]))
	r := C.Predict(p.ctx, ptr)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}
	return predictions, nil
}

func (p *Predictor) Close() {
	C.Delete(p.ctx)
}

func SetUseCPU() {
	C.SetMode(C.int(CPUMode))
}

func SetUseGPU() {
	C.SetMode(C.int(GPUMode))
}

func init() {
	C.Init()
}
