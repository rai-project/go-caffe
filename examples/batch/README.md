# Batch Image Classification with Tracing

## jaeger
```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
812bba651a4a1c5a5d3c5eac5de610759bf54f716d1e531017b4e206b964e1e8
```
