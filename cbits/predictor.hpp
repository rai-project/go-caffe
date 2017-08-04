#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef struct PredictorContext PredictorContext;

PredictorContext *New(char *model_file, char *trained_file, char *mean_file,
                      char *label_file);

const char *Predict(PredictorContext *pred, char *buffer, size_t length);

void Delete(PredictorContext *pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
