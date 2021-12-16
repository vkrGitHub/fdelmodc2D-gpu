#ifndef _CUDA_MYUTILS_
#define _CUDA_MYUTILS_

#include "fdelmodc.h"

void printfgpu(char *fmt, ...);
void wrap_cudaGetLastError (const char *msg);
void wrap_cudaGetDeviceProperties();
void wrap_cudaMemGetInfo(char *msg);
// void cuda_printWhichGPU2(char *text);
// void cuda_printWhichGPU(char *fmt, ...);
void cuda_wrap_fread(void *d_input, int size, int stride, char *filename);
void cuda_wrap_fwrite(void *d_input, int size, int stride, char *filename);
void cuda_wrap_fwrite_append(void *d_input, int size, int stride, char *filename);

void wrap_fread(void *input, int size, int stride, char *filename);
void wrap_fwrite(void *input, int size, int stride, char *filename);
void wrap_fwrite_append(void *input, int size, int stride, char *filename);


#endif

