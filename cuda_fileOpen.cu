#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

extern "C"{
	#include <assert.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <errno.h>
	#include <math.h>
	#include <string.h>
	#include "segy.h"
}

/**
*  File handling routines 
*
*   AUTHOR:
*           Jan Thorbecke (janth@xs4all.nl)
*           The Netherlands 
**/

void name_ext(char *filename, char *extension)
{
	char ext[100];

	if (strstr(filename, ".su") != NULL) {
		sprintf(ext,"%s.su", extension);
		strcpy(strstr(filename, ".su"), ext);
	}
	else if (strstr(filename, ".segy") != NULL) {
		sprintf(ext,"%s.segy", extension);
		strcpy(strstr(filename, ".segy"), ext);
	}
	else if (strstr(filename, ".mat") != NULL) {
		sprintf(ext,"%s.mat", extension);
		strcpy(strstr(filename, ".mat"), ext);
	}
	else if (strstr(filename, ".hdf") != NULL) {
		sprintf(ext,"%s.hdf", extension);
		strcpy(strstr(filename, ".hdf"), ext);
	}
	else if (strrchr(filename, '.') != NULL) {
		sprintf(ext,"%s.su", extension);
		strcpy(strrchr(filename, '.'), ext);
	}
	else {
		sprintf(ext,"%s.su", extension);
		strcat(filename, ext);
	}

	return;
}

FILE *fileOpen(char *file, char *ext, int append)
{
	FILE *fp;
	char filename[1024];

	strcpy(filename, file);
	name_ext(filename, ext);
	if (append) fp = fopen(filename, "a");
   	else fp = fopen(filename, "w");
   	assert(fp != NULL);
	
	return fp;
}

int traceWrite(segy *hdr, float *data, int n, FILE *fp) 
{
    size_t  nwrite;

    nwrite = fwrite( hdr, 1, TRCBYTES, fp);
    assert(nwrite == TRCBYTES);
    nwrite = fwrite( data, sizeof(float), n, fp);
    assert(nwrite == n);

	return 0;
}



int cuda_traceWrite(segy *hdr, float *d_data, int n, cudaStream_t stream, FILE *fp) 
{
    size_t  nwrite;
    float data[n];

	cudaMemcpyAsync(data, d_data, n*sizeof(float), cudaMemcpyDeviceToHost, stream);
	
    nwrite = fwrite( hdr, 1, TRCBYTES, fp);
    assert(nwrite == TRCBYTES);
    nwrite = fwrite( data, sizeof(float), n, fp);
    assert(nwrite == n);

    return 0;
}

