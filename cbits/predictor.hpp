#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorHandle;

PredictorHandle NewCaffe(char *model_file, char *trained_file, int batch_size,
                         int mode);

void SetModeCaffe(int mode);

void InitCaffe();

void PredictCaffe(PredictorHandle pred);

void SetInputCaffe(PredictorHandle pred, int idx, float *data);

const float *GetOutputDataCaffe(PredictorHandle pred, int idx);

const int *GetOutputShapeCaffe(PredictorHandle pred, int idx, int *len);

void DeleteCaffe(PredictorHandle pred);

void StartProfilingCaffe(PredictorHandle pred, const char *name,
                         const char *metadata);

void EndProfilingCaffe(PredictorHandle pred);

void DisableProfilingCaffe(PredictorHandle pred);

char *ReadProfileCaffe(PredictorHandle pred);

int GetWidthCaffe(PredictorHandle pred);

int GetHeightCaffe(PredictorHandle pred);

int GetChannelsCaffe(PredictorHandle pred);

// int GetPredLenCaffe(PredictorHandle pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
