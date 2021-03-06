package caffe

import (
	"io/ioutil"

	"github.com/Unknwon/com"
	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	caffeproto "github.com/rai-project/go-caffe/proto"
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

func ReadBlob(protoFileName string) (*caffeproto.BlobProto, error) {
	blob := new(caffeproto.BlobProto)
	err := unmarshal(blob, protoFileName)
	return blob, err
}
