package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system -L/usr/local/lib
// #cgo CXXFLAGS: -std=c++14 -I/usr/local/include/ -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1
// #cgo darwin CXXFLAGS: -DBLAS=open -I/usr/local/opt/openblas/include -I/opt/caffe/include
// #cgo darwin LDFLAGS: -L/usr/local/opt/openblas/lib -L/opt/caffe/lib
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
	e := SolverParameter_SolverMode_value["CPU"]
	C.SetMode(C.int(e))
}

func SetUseGPU() {
	e := SolverParameter_SolverMode_value["GPU"]
	C.SetMode(C.int(e))
}
