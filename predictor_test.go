package caffe

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	caffeModelFile = "./_fixtures/cifar10_nin.caffemodel"
	trainVal       = "./_fixtures/train_val.prototxt"
)

func TestCreatePredictor(t *testing.T) {
	predictor, err := New(caffeModelFile, trainVal)
	assert.NoError(t, err)
	defer predictor.Close()
	assert.NotEmpty(t, predictor)
}
