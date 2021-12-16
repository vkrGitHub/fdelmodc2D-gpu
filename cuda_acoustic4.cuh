#ifndef _ACOUSTIC4_CUDA_CUH
#define _ACOUSTIC4_CUDA_CUH

extern "C"{
	#include <stdio.h>
	#include <stdlib.h>
	#include "fdelmodc.h"
}


void cuda_init_acoustic4(modPar mod, int verbose);
void cuda_destroy_acoustic4(int verbose);

void cuda_run_acoustic4(modPar    mod, srcPar    src, wavPar    wav, bndPar    bnd, 
						modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, int itime, int ixsrc, int izsrc, float *d_src_nwav, float *d_vx, float *d_vz, float *d_p, float *d_rox, float *d_roz, float *d_l2m, int verbose);
void cuda_run_acoustic42(modPar   mod, srcPar    src, wavPar    wav, bndPar    bnd, 
						modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, int itime, int ixsrc, int izsrc, float *d_src_nwav, float *d_vx, float *d_vz, float *d_p, float *d_rox, float *d_roz, float *d_l2m, int verbose);

void cuda_print_acoustic4_time(int verbose);

#endif









