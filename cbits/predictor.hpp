#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext New(char *model_file, char *trained_file, unsigned int batch);

const char *Predict(PredictorContext pred, float *imageData);

int PredictorGetChannel(PredictorContext pred);

int PredictorGetWidth(PredictorContext pred);

int PredictorGetHeight(PredictorContext pred);

int PredictorGetBatchSize(PredictorContext pred);

void Delete(PredictorContext pred);

void SetMode(int mode);

void Init();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
