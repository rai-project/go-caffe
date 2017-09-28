// Created by cgo - DO NOT EDIT

package caffe

import "unsafe"

import _ "runtime/cgo"

import "syscall"

var _ syscall.Errno
func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }

//go:linkname _Cgo_always_false runtime.cgoAlwaysFalse
var _Cgo_always_false bool
//go:linkname _Cgo_use runtime.cgoUse
func _Cgo_use(interface{})
type _Ctype_PredictorContext unsafe.Pointer

type _Ctype_char int8

type _Ctype_void [0]byte

//go:linkname _cgo_runtime_cgocall runtime.cgocall
func _cgo_runtime_cgocall(unsafe.Pointer, uintptr) int32

//go:linkname _cgo_runtime_cgocallback runtime.cgocallback
func _cgo_runtime_cgocallback(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr)

//go:linkname _cgoCheckPointer runtime.cgoCheckPointer
func _cgoCheckPointer(interface{}, ...interface{})

//go:linkname _cgoCheckResult runtime.cgoCheckResult
func _cgoCheckResult(interface{})

//go:cgo_import_static _cgo_7c15b8eb7ede_Cfunc_Delete
//go:linkname __cgofn__cgo_7c15b8eb7ede_Cfunc_Delete _cgo_7c15b8eb7ede_Cfunc_Delete
var __cgofn__cgo_7c15b8eb7ede_Cfunc_Delete byte
var _cgo_7c15b8eb7ede_Cfunc_Delete = unsafe.Pointer(&__cgofn__cgo_7c15b8eb7ede_Cfunc_Delete)

//go:cgo_unsafe_args
func _Cfunc_Delete(p0 *_Ctype_PredictorContext) (r1 _Ctype_void) {
	_cgo_runtime_cgocall(_cgo_7c15b8eb7ede_Cfunc_Delete, uintptr(unsafe.Pointer(&p0)))
	if _Cgo_always_false {
		_Cgo_use(p0)
	}
	return
}
//go:cgo_import_static _cgo_7c15b8eb7ede_Cfunc_New
//go:linkname __cgofn__cgo_7c15b8eb7ede_Cfunc_New _cgo_7c15b8eb7ede_Cfunc_New
var __cgofn__cgo_7c15b8eb7ede_Cfunc_New byte
var _cgo_7c15b8eb7ede_Cfunc_New = unsafe.Pointer(&__cgofn__cgo_7c15b8eb7ede_Cfunc_New)

//go:cgo_unsafe_args
func _Cfunc_New(p0 *_Ctype_char, p1 *_Ctype_char) (r1 *_Ctype_PredictorContext) {
	_cgo_runtime_cgocall(_cgo_7c15b8eb7ede_Cfunc_New, uintptr(unsafe.Pointer(&p0)))
	if _Cgo_always_false {
		_Cgo_use(p0)
		_Cgo_use(p1)
	}
	return
}
