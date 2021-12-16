#ifndef _CUDA_SOURCEONSURFACE_
#define _CUDA_SOURCEONSURFACE_

#include"fdelmodc.h"

void cuda_allocStoreSourceOnSurface(srcPar src, int verbose);
void cuda_freeStoreSourceOnSurface(int verbose);
__global__ void kernel_storeSourceOnSurface(modPar *d_mod, srcPar *d_src, bndPar *d_bnd, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, int verbose);
__global__ void kernel_reStoreSourceOnSurface(modPar *d_mod, srcPar *d_src, bndPar *d_bnd, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, int verbose);




#endif
