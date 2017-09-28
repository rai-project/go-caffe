// Created by cgo - DO NOT EDIT

//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:1
package caffe
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:10

//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:9
import "github.com/Unknwon/com"
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:12

//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:11
type Predictor _Ctype_PredictorContext
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:14

//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:13
func NewPredictor(modelFile, trainFile string) (*Predictor, error) {
	if !com.IsFile(modelFile) {
		return nil, errors.Errrof("file %s not found", modelFile)
	}
	if !com.IsFile(trainFile) {
		return nil, errors.Errrof("file %s not found", trainFile)
	}
	return _Cfunc_New(modelFile, trainFile), nil
}
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:24

//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:23
func (p *Predictor) Close() {
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:23
	func(_cgo0 *_Ctype_PredictorContext) {
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:23
		_cgoCheckPointer(_cgo0)
														_Cfunc_Delete(_cgo0)
//line /Users/abduld/.gvm/pkgsets/go1.8.1/global/src/github.com/rai-project/go-caffe/cbits.go:24
	}(p.PredictorContext)
}
