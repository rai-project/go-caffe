package caffe

import (
	"io/ioutil"

	"github.com/Unknwon/com"
	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
)

func unmarshal(target proto.Message, protoFileName string) error {
	if !com.IsFile(protoFileName) {
		return errors.Errorf("%s is not a file", protoFileName)
	}
	buf, err := ioutil.ReadFile(protoFileName)
	if err != nil {
		return errors.Wrapf(err, "failed to open %s", protoFileName)
	}

	return proto.Unmarshal(buf, target)
}

func ReadBlob(protoFileName string) (*BlobProto, error) {
	blob := new(BlobProto)
	err := unmarshal(blob, protoFileName)
	return blob, err
}
