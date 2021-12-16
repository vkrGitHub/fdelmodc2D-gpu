#ifndef _CUDA_WRITEREC_CUH
#define _CUDA_WRITEREC_CUH

#include "fdelmodc.h"

void cuda_init_writeRecTimes();
void cuda_destroy_writeRecTimes();

int cuda_writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav, int ixsrc, int izsrc, int nsam, int ishot, int fileno, 
             float *d_rec_vx, float *d_rec_vz, float *d_rec_txx, float *d_rec_tzz, float *d_rec_txz, 
             float *d_rec_p, float *d_rec_pp, float *d_rec_ss, float *d_rec_udp, float *d_rec_udvz, int verbose);

void cuda_print_writeRecTimes_time();

#endif