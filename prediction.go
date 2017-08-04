package caffe

type Prediction struct {
	Index       int     `json:"index"`
	Probability float32 `json:"probability"`
}
