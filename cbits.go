package caffe

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/rai-project/nvidia-smi"

	"github.com/Unknwon/com"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
)

const (
	CPUMode = 0
	GPUMode = 1
)

type Predictor struct {
	ctx     C.PredictorContext
	options *options.Options
}

func New(opts ...options.Option) (*Predictor, error) {
	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}
	weightsFile := string(options.Weights())
	if !com.IsFile(weightsFile) {
		return nil, errors.Errorf("file %s not found", weightsFile)
	}

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
		ctx: C.CaffeNew(
			C.CString(modelFile),
			C.CString(weightsFile),
			C.int(options.BatchSize()),
			C.int(mode),
		),
		options: options,
	}, nil
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.CaffeStartProfiling(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.CaffeEndProfiling(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.CaffeDisableProfiling(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.CaffeReadProfile(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}

func (p *Predictor) Predict(ctx context.Context, data []float32) (Predictions, error) {
	// check input
	if data == nil || len(data) < 1 {
		return nil, fmt.Errorf("intput data nil or empty")
	}

	span := opentracing.SpanFromContext(ctx)
	span.LogFields(
		olog.String("event", "before caffe padding"),
	)

	batchSize := int64(C.CaffePredictorGetBatchSize(p.ctx))
	if batchSize != 1 {
		width := C.CaffePredictorGetWidth(p.ctx)
		height := C.CaffePredictorGetHeight(p.ctx)
		channels := C.CaffePredictorGetChannels(p.ctx)

		dataLen := int64(len(data))
		shapeLen := int64(width * height * channels)
		inputCount := dataLen / shapeLen
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	span.LogFields(
		olog.String("event", "after caffe padding"),
	)

	ptr := (*C.float)(unsafe.Pointer(&data[0]))
	r := C.CaffePredict(p.ctx, ptr)
	defer C.free(unsafe.Pointer(r))
	js := C.GoString(r)

	span.LogFields(
		olog.String("event", "before caffe unmarshal"),
	)

	predictions := []Prediction{}
	err := json.Unmarshal([]byte(js), &predictions)
	if err != nil {
		return nil, err
	}

	span.LogFields(
		olog.String("event", "after caffe unmarshal"),
	)

	return predictions, nil
}

func (p *Predictor) Close() {
	C.CaffeDelete(p.ctx)
}

func SetUseCPU() {
	pp.Println("Setting to use CPU")
	C.CaffeSetMode(C.int(CPUMode))
}

func SetUseGPU() {
	C.CaffeSetMode(C.int(GPUMode))
}

func init() {
	C.CaffeInit()
}
