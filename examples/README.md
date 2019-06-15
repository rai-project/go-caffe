# Caffe Image Classification in Go with Profiling

The two examples show how to run caffe image classificaiton models in Go with nvprof and MLModelScope profiling.

## With nvprof

```
cd batch
go build
nvprof --profile-from-start off ./batch_nvprof
```

## With jaeger

Start jaeger docker container by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
812bba651a4a1c5a5d3c5eac5de610759bf54f716d1e531017b4e206b964e1e8
```

Then run the example by

```
cd batch
go run main.go
```
Go to ```localhost:16686``` to see the trace


