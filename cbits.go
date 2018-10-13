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
	ctx     C.PredictorContext
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
		ctx: C.NewCaffe(
			C.CString(modelFile),
			C.CString(weightsFile),
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

func (p *Predictor) Predict(ctx context.Context, data []float32) error {
	if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

	batchSize := p.options.BatchSize()
	width := C.PredictorGetWidthCaffe(p.ctx)
	height := C.PredictorGetHeightCaffe(p.ctx)
	channels := C.PredictorGetChannelsCaffe(p.ctx)
	shapeLen := int(width * height * channels)
	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.PredictCaffe(p.ctx, ptr)

	return nil
}

func (p *Predictor) ReadPredictedFeatures(ctx context.Context) Predictions {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "read_predicted_features")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.PredictorGetPredLenCaffe(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsCaffe(p.ctx)

	slice := (*[1 << 30]C.float)(unsafe.Pointer(cPredictions))[:length:length]

	predictions := make([]Prediction, length)
	for ii := 0; ii < length; ii++ {
		predictions[ii] = Prediction{
			Index:       ii % predLen,
			Probability: float32(slice[ii]),
		}
	}

	return predictions
}

func (p *Predictor) Close() {
	C.DeleteCaffe(p.ctx)
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingCaffe(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingCaffe(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingCaffe(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileCaffe(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
