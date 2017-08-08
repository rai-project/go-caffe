package caffe

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthonynsimon/bild/parallel"
	"github.com/k0kubun/pp"

	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	homedir "github.com/mitchellh/go-homedir"
	"github.com/stretchr/testify/assert"
)

var (
	// caffeModelFile = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "cifar10_nin.caffemodel")
	// trainVal       = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "nin_model.prototxt")
	homeDir, _         = homedir.Dir()
	caffeModelURL      = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel"
	trailValURL        = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt"
	caffeModelFileName = filepath.Join(homeDir, "Downloads", "bvlc_reference_caffenet.caffemodel")
	trainValFileName   = filepath.Join(homeDir, "Downloads", "deploy.prototxt")
	imageFileName      = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "chicken.jpg")
	meanImage          = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "imagenet_mean.binaryproto")
)

func getImageData(t *testing.T, img image.Image) ([]float32, error) {

	b := img.Bounds()
	height := b.Max.Y - b.Min.Y // image height
	width := b.Max.X - b.Min.X  // image width

	meanBlob, err := ReadBlob(meanImage)
	assert.NoError(t, err)

	meanData := meanBlob.Data
	mean := [3]float32{}
	for cc := 0; cc < 3; cc++ {
		accum := float32(0)
		offset := cc * width * height
		for ii := 0; ii < height; ii++ {
			for jj := 0; jj < width; jj++ {
				accum += meanData[offset+ii*width+jj]
			}
		}
		mean[cc] = accum / float32(width*height)
	}

	res := make([]float32, 3*height*width)
	parallel.Line(height, func(start, end int) {
		w := width
		h := height
		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
				res[y*w+x] = float32(r>>8) - mean[2]
				res[w*h+y*w+x] = float32(g>>8) - mean[1]
				res[2*w*h+y*w+x] = float32(b>>8) - mean[0]
			}
		}
	})

	return res, nil
}

func TestCreatePredictor(t *testing.T) {
	SetUseCPU()
	predictor, err := New(trainValFileName, caffeModelFileName)
	assert.NoError(t, err)
	defer predictor.Close()
	assert.NotEmpty(t, predictor)

	imgFile, err := os.Open(imageFileName)
	assert.NoError(t, err)
	defer imgFile.Close()

	image, _, err := image.Decode(imgFile)
	assert.NoError(t, err)
	assert.NotEmpty(t, image)

	// caffenet dim is 227
	imageWidth, imageHeight := image.Bounds().Dx(), image.Bounds().Dy()
	assert.Equal(t, 227, imageWidth)
	assert.Equal(t, 227, imageHeight)

	imageData, err := getImageData(t, image)
	assert.NoError(t, err)

	predictions, err := predictor.Predict(imageData)
	predictions.Sort()
	pp.Println(predictions[0:2])

	assert.NoError(t, err)
	assert.NotEmpty(t, predictions)
	assert.Equal(t, 1000, len(predictions))
}

func init() {
	SetUseCPU()
}
