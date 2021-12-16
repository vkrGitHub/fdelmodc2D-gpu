#ifndef _CUDA_WRITESNAPTIMES_CUH_
#define _CUDA_WRITESNAPTIMES_CUH_

extern "C"{
	#include "fdelmodc.h"		
}

// void cuda_init_writeSnapTimes(snaPar sna);
// void cuda_destroy_writeSnapTimes();

void cuda_writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav, int ixsrc, int izsrc, int itime, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, int verbose);



#endif