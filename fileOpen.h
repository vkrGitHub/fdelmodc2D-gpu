#ifndef _FILEOPEN_H_
#define _FILEOPEN_H_

#include "segy.h"

void name_ext(char *filename, char *extension);
FILE *fileOpen(char *file, char *ext, int append);
int traceWrite(segy *hdr, float *data, int n, FILE *fp);


#endif