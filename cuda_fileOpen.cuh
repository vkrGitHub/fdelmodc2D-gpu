#ifndef _FILEOPEN_CUH_
#define _FILEOPEN_CUH_

#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

extern "C"{
	#include "segy.h"
}

void name_ext(char *filename, char *extension);
FILE *fileOpen(char *file, char *ext, int append);
int traceWrite(segy *hdr, float *data, int n, FILE *fp);
int cuda_traceWrite(segy *hdr, float *d_data, int n, cudaStream_t stream, FILE *fp);


#endif
