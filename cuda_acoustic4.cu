extern "C"{
	#include<stdlib.h>
	#include<stdio.h>
	#include<string.h>
	#include "fdelmodc.h"	
} 

#include "cuda_myutils.cuh"
#include "cuda_applySource.cuh"
#include "cuda_boundaries.cuh"
#include "cuda_fdel2d_parameters.cuh" //del later // for deepcpy tmp test
#include "cuda_sourceOnSurface.cuh"

// GPU timers
static cudaEvent_t t0, t1;
static cudaEvent_t marks[10];
static float times[9];
static float tmsec = 0; 
static float tall = 0; 
static int nrun = 0;

// Functions
void cuda_init_acoustic4(modPar mod, int verbose){
/*
Copies constants to CUDA symbol memory (faster access)
*/

	// Create timers
	cudaEventCreate(&t0); 
	cudaEventCreate(&t1); 
	for(int i=0; i<10; i++){
		cudaEventCreate(&marks[i]);
	}
	memset(times, 0, 9*sizeof(float));

	// Output to user
	if(verbose>1) {
		// printfgpu("cuda_init_acoustic4.");
		printfgpu("cuda_init_acoustic4.");
	}
}

void cuda_destroy_acoustic4(int verbose){
	// Destroy timers
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	for(int i=0; i<9; i++){
		cudaEventDestroy(marks[i]);
	}

	// Output to user
	if(verbose>1) printfgpu("cuda_destroy_acoustic4.");//uncomment
}


__global__ void kernel_acoustic4_vx(modPar *d_mod, float *d_p, float *d_rox, float *d_vx){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	float c1 = 9.0/8.0; 
	float c2 = -1.0/24.0;

	int n1 = d_mod->naz;
	int ixo = d_mod->ioXx;
	int ixe = d_mod->ieXx;
	int izo = d_mod->ioXz;
	int ize = d_mod->ieXz;

	int idz = id%n1;
	int idx = id/n1;

	/* calculate vx for all grid points except on the virtual boundary */
	if( (idx >= ixo) && (idx < ixe) && (idz >= izo) && (idz < ize) ){
			d_vx[idx*n1+idz] -= d_rox[idx*n1+idz]*(
						c1*(d_p[idx*n1+idz]   - d_p[(idx-1)*n1+idz]) +
						c2*(d_p[(idx+1)*n1+idz] - d_p[(idx-2)*n1+idz]));
	}
	
}

__global__ void kernel_acoustic4_vz(modPar *d_mod, float *d_p, float *d_roz, float *d_vz){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	float c1 = 9.0/8.0; 
	float c2 = -1.0/24.0;

	int n1 = d_mod->naz;
	int ixo = d_mod->ioZx;
	int ixe = d_mod->ieZx;
	int izo = d_mod->ioZz;
	int ize = d_mod->ieZz;

	int idz = id%n1;
	int idx = id/n1;

	/* calculate vz for all grid points except on the virtual boundary */
		if( (idx >= ixo) && (idx < ixe) && (idz >= izo) && (idz < ize) ){
			d_vz[idx*n1+idz] -= d_roz[idx*n1+idz]*(
						c1*(d_p[idx*n1+idz]   - d_p[idx*n1+idz-1]) +
						c2*(d_p[idx*n1+idz+1] - d_p[idx*n1+idz-2]));
	}
	
}


__global__ void kernel_acoustic4_p(modPar *d_mod, bndPar *d_bnd, float *d_vx, float *d_vz, float *d_l2m, float *d_p){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	float c1 = 9.0/8.0; 
	float c2 = -1.0/24.0;

	int n1 = d_mod->naz;
	int ixo = d_mod->ioPx;
	int ixe = d_mod->iePx;
	int izo = d_mod->ioPz;
	int ize = d_mod->iePz;

	if (d_bnd->lef==2) ixo += d_bnd->npml;
    if (d_bnd->rig==2) ixe -= d_bnd->npml;
    if (d_bnd->top==2) izo += d_bnd->npml;
    if (d_bnd->bot==2) ize -= d_bnd->npml;

	int idz = id%n1;
	int idx = id/n1;

	/* calculate p for all grid points except on the virtual boundary */
	if( (idx >= ixo) && (idx < ixe) && (idz >= izo) && (idz < ize) ){
			d_p[idx*n1+idz] -= d_l2m[idx*n1+idz]*(
						c1*(d_vx[(idx+1)*n1+idz] - d_vx[idx*n1+idz]) +
						c2*(d_vx[(idx+2)*n1+idz] - d_vx[(idx-1)*n1+idz]) +
						c1*(d_vz[idx*n1+idz+1]   - d_vz[idx*n1+idz]) +
						c2*(d_vz[idx*n1+idz+2]   - d_vz[idx*n1+idz-1]));
	}
	
}


//t acoustic4          (modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose)
void cuda_run_acoustic4(modPar    mod, srcPar    src, wavPar    wav, bndPar    bnd, 
						modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, int itime, int ixsrc, int izsrc, float *d_src_nwav, float *d_vx, float *d_vz, float *d_p, float *d_rox, float *d_roz, float *d_l2m, int verbose)
{
	int ixo, ixe, izo, ize;

	// Time 
	cudaEventRecord(t0); 
	cudaEventRecord(marks[0]); 

	kernel_acoustic4_vx<<<(mod.naz*mod.nax+255)/256, 256>>>(d_mod, d_p, d_rox, d_vx);
	wrap_cudaGetLastError("after kernel_acoustic4_vx");
	cudaEventRecord(marks[1]); 
	cudaEventSynchronize(marks[1]); 
	cudaEventElapsedTime(&tmsec, marks[0], marks[1]);
	times[0] += tmsec*1E-3;

	kernel_acoustic4_vz<<<(mod.naz*mod.nax+255)/256, 256>>>(d_mod, d_p, d_roz, d_vz);
	wrap_cudaGetLastError("after kernel_acoustic4_vz");
	cudaEventRecord(marks[2]); 
	cudaEventSynchronize(marks[2]); 
	cudaEventElapsedTime(&tmsec, marks[1], marks[2]);
	times[1] += tmsec*1E-3;

	// boundariesP(              mod,   bnd,   vx,   vz,     p,  NULL,  NULL,   rox,   roz,   l2m,  NULL,  NULL, itime, verbose);
	cuda_boundariesP(mod, bnd, d_mod, d_bnd, d_vx, d_vz,   d_p,  NULL,  NULL, d_rox, d_roz, d_l2m,  NULL,  NULL, itime, verbose);
	wrap_cudaGetLastError("after cuda_boundariesP");
	cudaEventRecord(marks[3]); 
	cudaEventSynchronize(marks[3]); 
	cudaEventElapsedTime(&tmsec, marks[2], marks[3]);
	times[2] += tmsec*1E-3;

	if (src.type > 5) {
	// 	 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, p, NULL, NULL, rox, roz, l2m, src_nwav, verbose);
		kernel_applySource<<<1,1>>>(d_mod, d_src, d_wav, d_bnd, itime, ixsrc, izsrc, d_vx, d_vz,   d_p,  NULL,  NULL, d_rox, d_roz, d_l2m, d_src_nwav, verbose);
	}
	wrap_cudaGetLastError("after kernel_applySource (src.type>5)");
	cudaEventRecord(marks[4]); 
	cudaEventSynchronize(marks[4]); 
	cudaEventElapsedTime(&tmsec, marks[3], marks[4]);
	times[3] += tmsec*1E-3;

	//kernel_acoustic4_p<<<(n1*n2+255)/256, 256>>>(ixo, ixe, izo, ize, d_vx, d_vz, d_l2m, d_p);
	kernel_acoustic4_p<<<(mod.naz*mod.nax+255)/256, 256>>>(d_mod, d_bnd, d_vx, d_vz, d_l2m, d_p);
	wrap_cudaGetLastError("after kernel_acoustic4_p");
	cudaEventRecord(marks[5]); 
	cudaEventSynchronize(marks[5]); 
	cudaEventElapsedTime(&tmsec, marks[4], marks[5]);
	times[4] += tmsec*1E-3;

	if (src.type < 6) {
		kernel_applySource<<<1,1>>>(d_mod, d_src, d_wav, d_bnd, itime, ixsrc, izsrc, d_vx, d_vz,   d_p,  NULL,  NULL, d_rox, d_roz, d_l2m, d_src_nwav, verbose);
	}
	wrap_cudaGetLastError("after kernel_applySource");
	cudaEventRecord(marks[6]); 
	cudaEventSynchronize(marks[6]); 
	cudaEventElapsedTime(&tmsec, marks[5], marks[6]);
	times[5] += tmsec*1E-3;

	// Free surface and PML
	/* check if there are sources placed on the free surface */
    //storeSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, p, NULL, NULL, verbose);
	kernel_storeSourceOnSurface<<<1,1>>>(d_mod, d_src, d_bnd, ixsrc, izsrc, d_vx, d_vz, d_p, NULL, NULL, verbose);
	wrap_cudaGetLastError("after kernel_storeSourceOnSurface");
	cudaEventRecord(marks[7]); 
	cudaEventSynchronize(marks[7]); 
	cudaEventElapsedTime(&tmsec, marks[6], marks[7]);
	times[6] += tmsec*1E-3;

	/* Free surface: calculate free surface conditions for stresses */
	//boundariesV(   mod, bnd,                 vx,   vz,    p,  NULL,  NULL,   rox,   roz,   l2m,  NULL,  NULL, itime, verbose);
	cuda_boundariesV(mod, bnd, d_mod, d_bnd, d_vx, d_vz,  d_p,  NULL,  NULL, d_rox, d_roz, d_l2m,  NULL,  NULL, itime, verbose);
	wrap_cudaGetLastError("after cuda_boundariesV");
	cudaEventRecord(marks[8]); 
	cudaEventSynchronize(marks[8]); 
	cudaEventElapsedTime(&tmsec, marks[7], marks[8]);
	times[7] += tmsec*1E-3;

	/* restore source positions on the edge */
	//reStoreSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, p, NULL, NULL, verbose);
	kernel_reStoreSourceOnSurface<<<1,1>>>(d_mod, d_src, d_bnd, ixsrc, izsrc, d_vx, d_vz, d_p, NULL, NULL, verbose);
	wrap_cudaGetLastError("after kernel_reStoreSourceOnSurface");
	cudaEventRecord(marks[9]); 
	cudaEventSynchronize(marks[9]); 
	cudaEventElapsedTime(&tmsec, marks[8], marks[9]);
	times[8] += tmsec*1E-3;

	// cudaEventRecord(t1); 
	// cudaEventSynchronize(t1); 
	cudaEventElapsedTime(&tmsec, marks[0], marks[9]);
	tall += tmsec*1E-3; \
	nrun ++;

	wrap_cudaGetLastError("after kernel_reStoreSourceOnSurface");
}

void cuda_print_acoustic4_time(int verb){
	printf("--------------- cuda_run_acoustic4 times: ---------------\n");
	if(verb>=1) printf("\tkernel_acoustic4_vx:\t %.4f s (%dx).\n", times[0], nrun);
	if(verb>=1) printf("\tkernel_acoustic4_vz:\t %.4f s (%dx).\n", times[1], nrun);
	if(verb>=1) printf("\tcuda_boundariesP:\t %.4f s (%dx).\n", times[2], nrun);
	if(verb>=1) printf("\tkernel_acoustic4_p:\t %.4f s (%dx).\n", times[4], nrun);
	if(verb>=1) printf("\tkernel_applySource:\t %.4f s (%dx).\n", times[5]+times[3], nrun);
	if(verb>=3) printf("\tkernel_(re)storeSourceOnSurface:\t %.4f s (%dx).\n", times[6]+times[8], nrun);
	if(verb>=1) printf("\tcuda_boundariesV:\t %.4f s (%dx).\n", times[7], nrun);

	if(verb>=1) printf("\tTotal:\t\t\t %.4f s (%dx).\n", tall, nrun);
	printf("---------------------------------------------------------\n");
}
