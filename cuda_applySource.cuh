#ifndef _CUDA_APPLYSOURCE_CUH_
#define _CUDA_APPLYSOURCE_CUH_

extern "C"{
	#include"fdelmodc.h"
}


//void applySource(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float **src_nwav, int verbose);
__global__ void kernel_applySource(modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, int itime, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_src_nwav, int verbose);


#endif



