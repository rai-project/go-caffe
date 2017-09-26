package caffe

// #cgo LDFLAGS: -lcaffe -lstdc++ -lglog -lboost_system -L/usr/local/lib  -L/opt/caffe/lib
// #cgo CXXFLAGS: -DBLAS=open -std=c++11 -I/usr/local/include/ -I${SRCDIR}/cbits -O3 -Wall -DCPU_ONLY=1 -I/opt/caffe/include
// #cgo darwin CXXFLAGS: -I/usr/local/opt/openblas/include
// #cgo darwin LDFLAGS: -L/usr/local/opt/openblas/lib
// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"encoding/json"
	"fmt"
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

func New(modelFile, trainFile string, batch uint32) (*Predictor, error) {
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	if !com.IsFile(trainFile) {
		return nil, errors.Errorf("file %s not found", trainFile)
	}
	return &Predictor{
		ctx: C.New(C.CString(modelFile), C.CString(trainFile), C.uint(batch)),
	}, nil
}

func (p *Predictor) Predict(data []float32) (Predictions, error) {
	// check input
	if data == nil || len(data) < 1 {
		return nil, fmt.Errorf("intput data nil or empty")
	}

	batchSize := C.PredictorGetBatchSize(p.ctx)
	if batchSize != 1 {
		width := C.PredictorGetWidth(p.ctx)
		height := C.PredictorGetHeight(p.ctx)
		channel := C.PredictorGetChannel(p.ctx)

		dataLen := int64(len(data))
		shapeLen := int64(width * height * channel)
		inputCount := dataLen / shapeLen
		padding := make([]float32, (int64(batchSize)-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))
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
