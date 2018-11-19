UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
	SED='gsed'
else
	SED="sed"
endif

all: generate

fmt:
	go fmt ./...

install-deps:
	go get -u -v google.golang.org/grpc
	go get -u -v github.com/gogo/protobuf/proto
	go get -u -v github.com/gogo/protobuf/gogoproto
	go get -u -v github.com/golang/protobuf/protoc-gen-go
	go get -u -v github.com/gogo/protobuf/protoc-gen-gofast
	go get -u -v github.com/gogo/protobuf/protoc-gen-gogofaster
	go get -u -v github.com/gogo/protobuf/protoc-gen-gogoslick
	go get github.com/golang/dep
	dep ensure -v

generate: clean
	protoc --gogofaster_out=import_path=proto:proto -Iproto -I$(GOPATH)/src proto/caffe.proto
	${SED} -i '0,/func init/ s/func init/func disabled_init1/' proto/caffe.pb.go
	${SED} -i '0,/func init/ s/func init/func disabled_init2/' proto/caffe.pb.go
	${SED} -i '0,/func init/ s/func init/func disabled_init3/' proto/caffe.pb.go
	${SED} -i '0,/func init/ s/func init/func disabled_init4/' proto/caffe.pb.go
	${SED} -i '0,/func init/ s/func init/func disabled_init5/' proto/caffe.pb.go
	go fmt proto/...

clean-models:
	rm -fr builtin_models_static.go

clean:
	rm -fr proto/*pb.go
