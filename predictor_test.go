package caffe

import (
	"os"
	"path/filepath"
	"testing"

	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/anthonynsimon/bild/parallel"
	homedir "github.com/mitchellh/go-homedir"
	"github.com/stretchr/testify/assert"
)

var (
	// caffeModelFile = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "cifar10_nin.caffemodel")
	// trainVal       = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "nin_model.prototxt")
	homeDir, _         = homedir.Dir()
	caffeModelFileName = filepath.Join(homeDir, "Downloads", "bvlc_reference_caffenet.caffemodel")
	trainValFileName   = filepath.Join(homeDir, "Downloads", "deploy.txt")
	imageFileName      = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "banana.png")
)

func getImageData(img image.Image) ([]float32, error) {

	b := img.Bounds()
	height := b.Max.Y - b.Min.Y // image height
	width := b.Max.X - b.Min.X  // image width

	res := make([]float32, 3*height*width)
	parallel.Line(height, func(start, end int) {
		w := width
		h := height
		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
				res[y*w+x] = float32(r >> 8)
				res[w*h+y*w+x] = float32(g >> 8)
				res[2*w*h+y*w+x] = float32(b >> 8)
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

	imageWidth, imageHeight := image.Bounds().Dx(), image.Bounds().Dy()
	assert.Equal(t, 227, imageWidth)
	assert.Equal(t, 227, imageHeight)

	imageData, err := getImageData(image)
	assert.NoError(t, err)

	predictions, err := predictor.Predict(imageData)
	assert.NoError(t, err)
	assert.NotEmpty(t, predictions)
	assert.Equal(t, 1000, len(predictions))
}
