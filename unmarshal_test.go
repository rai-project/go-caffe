package caffe

import (
	"path/filepath"
	"testing"

	sourcepath "github.com/GeertJohan/go-sourcepath"
	"github.com/stretchr/testify/assert"
)

func TestUnmarshalModel(t *testing.T) {

	caffeModelFile := filepath.Join(sourcepath.MustAbsoluteDir(), "_fixtures", "imagenet_mean.binaryproto")

	blob, err := ReadBlob(caffeModelFile)
	assert.NoError(t, err)
	assert.NotEmpty(t, blob)

	assert.Equal(t, int32(256), blob.GetWidth())
	assert.Equal(t, int32(256), blob.GetHeight())
	assert.Equal(t, int32(3), blob.GetChannels())

}
