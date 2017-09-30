package caffe

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
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
		ctx: C.New(C.CString(modelFile), C.CString(trainFile), C.int(batch)),
	}, nil
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfiling(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfiling(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfiling(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfile(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func (p *Predictor) Predict(data []float32) (Predictions, error) {
	// check input
	if data == nil || len(data) < 1 {
		return nil, fmt.Errorf("intput data nil or empty")
	}

	batchSize := int64(C.PredictorGetBatchSize(p.ctx))
	if batchSize != 1 {
		width := C.PredictorGetWidth(p.ctx)
		height := C.PredictorGetHeight(p.ctx)
		channels := C.PredictorGetChannels(p.ctx)

		dataLen := int64(len(data))
		shapeLen := int64(width * height * channels)
		inputCount := dataLen / shapeLen
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}
	pp.Println(batchSize)

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
