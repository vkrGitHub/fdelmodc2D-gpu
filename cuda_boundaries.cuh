#ifndef _CUDA_BOUNDARIES_CUH_
#define _CUDA_BOUNDARIES_CUH_

extern "C"{
	#include <stdio.h>
	#include <stdlib.h>
	#include "fdelmodc.h"
}

void cuda_init_boundaries(modPar mod, bndPar bnd, int verbose);
//void cuda_init_boundaries(modPar mod);
void cuda_destroy_boundaries(bndPar bnd, int verbose);

void cuda_boundariesP(modPar mod, bndPar bnd, modPar *d_mod, bndPar *d_bnd, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul, int itime, int verbose);
void cuda_boundariesV(modPar mod, bndPar bnd, modPar *d_mod, bndPar *d_bnd, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul, int itime, int verbose);

#endif