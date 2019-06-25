package caffe

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/rai-project/tracer"
	"github.com/rai-project/nvidia-smi"
	"github.com/Unknwon/com"
	"github.com/pkg/errors"
  "github.com/rai-project/dlframework/framework/options"
)

const (
	CPUMode = 0
	GPUMode = 1
)

type Predictor struct {
  handle     C.PredictorHandle
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	weightsFile := string(options.Weights())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))

	weightsFileString := C.CString(weightsFile)
	defer C.free(unsafe.Pointer(weightsFileString))

	mode := CPUMode
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		SetUseGPU()
		mode = GPUMode
	} else {
		SetUseCPU()
	}

	return &Predictor{
		handle: C.NewCaffe(
			modelFileString,
			weightsFileString,
			C.int(options.BatchSize()),
			C.int(mode),
		),
		options: options,
	}, nil
}

func SetUseCPU() {
	C.SetModeCaffe(C.int(CPUMode))
}

func SetUseGPU() {
	C.SetModeCaffe(C.int(GPUMode))
}

func init() {
	C.InitCaffe()
}

// func (p *Predictor) Predict(ctx context.Context, data []tensor.Tensor) error {
func (p *Predictor) Predict(ctx context.Context) error {
  span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
  defer span.Finish()

	C.PredictCaffe(p.handle)

	return nil
}

func (p *Predictor) SetInput(idx int, data []float32) error {
  if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

  batchSize := p.options.BatchSize()
	width := C.GetWidthCaffe(p.handle)
	height := C.GetHeightCaffe(p.handle)
	channels := C.GetChannelsCaffe(p.handle)
	shapeLen := int(width * height * channels)
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))
  C.SetInputCaffe(p.handle, C.int(idx), ptr)

  return nil
}

func (p *Predictor) GetOutputShape(idx int) []int {
  var sz int32
  cdata := C.GetOutputShapeCaffe(p.handle, C.int32_t(idx), (*C.int32_t)(unsafe.Pointer(&sz)))
  res := make([]int, sz)
  data := (*[1 << 30]C.int32_t)(unsafe.Pointer(cdata))[:sz:sz]
  for ii := 0 ; ii< int(sz); ii++ {
    res[ii] = int(data[ii])
  }
  C.free(unsafe.Pointer(cdata))
	return res
}

func prod(sz []int) int {
  res := 1
  for _, a := range sz {
    res *= a
  }
  return res
}

func (p *Predictor) GetOutputData(idx int) []float32 {
  shape := p.GetOutputShape(idx)
  sz := prod(shape)

  data := C.GetOutputDataCaffe(p.handle, C.int32_t(idx))

	return (*[1 << 30]float32)(unsafe.Pointer(data))[:sz:sz]
}

func (p *Predictor) ReadOutputData(ctx context.Context, idx int ) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
  defer span.Finish()
  
	outputData :=p.GetOutputData(idx)
	if outputData == nil {
		return nil, errors.New("empty output data")
	}

	return outputData, nil
}

func (p *Predictor) Close() {
	C.DeleteCaffe(p.handle)
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingCaffe(p.handle, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingCaffe(p.handle)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingCaffe(p.handle)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileCaffe(p.handle)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
