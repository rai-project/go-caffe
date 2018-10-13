#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext NewCaffe(char *model_file, char *trained_file, int batch,
                          int mode);

void SetModeCaffe(int mode);

void InitCaffe();

void PredictCaffe(PredictorContext pred, float *imageData);

const float *GetPredictionsCaffe(PredictorContext pred);

void DeleteCaffe(PredictorContext pred);

void StartProfilingCaffe(PredictorContext pred, const char *name,
                         const char *metadata);

void EndProfilingCaffe(PredictorContext pred);

void DisableProfilingCaffe(PredictorContext pred);

char *ReadProfileCaffe(PredictorContext pred);

int GetWidthCaffe(PredictorContext pred);

int GetHeightCaffe(PredictorContext pred);

int GetChannelsCaffe(PredictorContext pred);

int GetPredLenCaffe(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
