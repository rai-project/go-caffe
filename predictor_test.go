package caffe

import (
	"path/filepath"
	"testing"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/stretchr/testify/assert"
)

var (
	caffeModelFile = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "cifar10_nin.caffemodel")
	trainVal       = filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "train_val.prototxt")
)

func XXTestCreatePredictor(t *testing.T) {
	predictor, err := New(caffeModelFile, trainVal)
	assert.NoError(t, err)
	defer predictor.Close()
	assert.NotEmpty(t, predictor)
}
