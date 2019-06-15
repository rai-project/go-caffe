package main

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"

	"github.com/Unknwon/com"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-caffe"
	cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
	"github.com/rai-project/tracer/ctimer"
)

var (
	batchSize   = 1
	model       = "bvlc_alexnet"
	shape       = []int{1, 3, 227, 227}
	mean        = []float32{123, 117, 104}
	scale       = []float32{1, 1, 1}
	imgDir, _   = filepath.Abs("../_fixtures")
	imgPath     = filepath.Join(imgDir, "platypus.jpg")
	graph_url   = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt"
	weights_url = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
	synset_url  = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32, scale []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.Bounds()
	height := in.Max.Y - in.Min.Y // image height
	width := in.Max.X - in.Min.X  // image width

	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := src.At(x+in.Min.X, y+in.Min.Y).RGBA()
			out[y*width+x] = (float32(b)/255 - mean[0]) / scale[0]
			out[width*height+y*width+x] = (float32(g)/255 - mean[1]) / scale[1]
			out[2*width*height+y*width+x] = (float32(r)/255 - mean[2]) / scale[2]
		}
	}

	return out, nil
}

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs("../tmp")
	dir = filepath.Join(dir, model)
	graph := filepath.Join(dir, "deploy.prototxt")
	weights := filepath.Join(dir, model+".caffemodel")
	synset := filepath.Join(dir, "synset.txt")

	if !com.IsFile(graph) {
		if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(weights) {
		if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(synset) {
		if _, err := downloadmanager.DownloadInto(synset_url, dir); err != nil {
			panic(err)
		}
	}

	img, err := imgio.Open(imgPath)
	if err != nil {
		panic(err)
	}

	height := shape[2]
	width := shape[3]

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, height, width, transform.Linear)
		res, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		caffe.SetUseGPU()
		device = options.CUDA_DEVICE
	} else {
		caffe.SetUseCPU()
	}

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "caffe_batch")
	defer span.Finish()

	predictor, err := caffe.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.Weights([]byte(weights)),
		options.BatchSize(batchSize))
	if err != nil {
		panic(err)
	}
	defer predictor.Close()

	err = predictor.SetInput(0, input)
	if err != nil {
		panic(err)
	}

	err = predictor.Predict(ctx)
	if err != nil {
		panic(err)
	}

	var cu *cupti.CUPTI
	if nvidiasmi.HasGPU {
		cu, err = cupti.New(cupti.Context(ctx))
		if err != nil {
			panic(err)
		}
	}

	predictor.StartProfiling("predict", "")

	err = predictor.Predict(ctx)
	if err != nil {
		panic(err)
	}

	predictor.EndProfiling()

	if nvidiasmi.HasGPU {
		cu.Wait()
		cu.Close()
	}

	profBuffer, err := predictor.ReadProfile()
	if err != nil {
		panic(err)
	}
	predictor.DisableProfiling()

	t, err := ctimer.New(profBuffer)
	if err != nil {
		panic(err)
	}
	t.Publish(ctx, tracer.FRAMEWORK_TRACE)

	output, err := predictor.ReadOutputData(ctx, 0)
	pp.Println(output[0:3])
	if err != nil {
		panic(err)
	}

	var labels []string
	f, err := os.Open(synset)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	if true {
		for i := 0; i < 1; i++ {
			results := features[i]
			top1 := results[0]
			pp.Println(top1.Probability)
			pp.Println(top1.GetClassification().GetLabel())
		}
	} else {
		_ = features
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
