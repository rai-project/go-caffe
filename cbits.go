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

	jsserializer "github.com/rai-project/serializer/json"

	"github.com/k0kubun/pp"
	"github.com/rai-project/nvidia-smi"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
)

const (
	CPUMode = 0
	GPUMode = 1
)

var (
	json = jsserializer.New()
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

func (p *Predictor) Predict(ctx context.Context, data []float32) error {
	if data == nil || len(data) < 1 {
		return nil, fmt.Errorf("intput data nil or empty")
	}

	batchSize := p.options.BatchSize()
	width := C.CaffePredictorGetWidth(p.ctx)
	height := C.CaffePredictorGetHeight(p.ctx)
	channels := C.CaffePredictorGetChannels(p.ctx)
	shapeLen := int(width * height * channels)
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "c_predict")

	res := C.CaffePredict(p.ctx, ptr)

	if predictSpan != nil {
		predictSpan.Finish()
	}

	return nil
}

func (p *Predictor) PostPredict(ctx context.Context) Predictions {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.STEP_TRACE, "post_predict")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.CaffePredictorGetPredLen(p.ctx))
	length := batchSize * predLen

	cPredictions := C.CaffeGetPredictions(p.ctx)
	slice := (*[1 << 30]C.float)(unsafe.Pointer(cPredictions))[:length:length]

	predictions := make([]Prediction, length)
	for ii := 0; ii < length; ii++ {
		predictions[ii] = Prediction{
			Index:       ii / batchSize,
			Probability: float32(slice[ii]),
		}
	}

	return predictions
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
