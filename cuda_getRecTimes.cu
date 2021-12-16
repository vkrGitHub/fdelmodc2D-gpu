extern "C"{
	#include<stdlib.h>
	#include<stdio.h>
	#include<math.h>
	#include<assert.h>
	#include"fdelmodc.h"
	#include"par.h"	
}

#include "cuda_myutils.cuh"

/**
*  Stores the wavefield at the receiver positions.
*
*  On a staggered grid the fields are all on different positions, 
*  to compensate for that the rec.int_vx and rec.int_vz options
*  can be set.
*
*   AUTHOR:
*           Jan Thorbecke (janth@xs4all.nl)
*           The Netherlands 
*
* GPU version can be optimized
* - Running kernels on different streams (better division)
**/

__constant__ int ibndx_d, ibndz_d; 
__constant__ float c1_d, c2_d;

// CUDA timers
static cudaEvent_t t0, t1;
static float tmsec = 0; 
static float tall = 0; 
static int nrun = 0;

// CUDA Streams for applying all interpolations simultaneously, if possible
static int nstream = 8; //for types p, txx, tzz, txz, pp, vz, vx, ud
static cudaStream_t *streams;

void cuda_init_getRecTimes(modPar mod, bndPar bnd){
/*
Init vars for cuda_getRecTimes
*/
	int n1, ibndx, ibndz;
	int irec, ix, iz, ix2, iz2, ix1, iz1;
	float dvx, dvz, rdz, rdx, C00, C10, C01, C11;
	float *vz_t, c1, c2, lroz, field;

    ibndx = mod.ioPx;
    ibndz = mod.ioPz;
    if (bnd.lef==4 || bnd.lef==2) ibndx += bnd.ntap;
    if (bnd.top==4 || bnd.top==2) ibndz += bnd.ntap;
	n1    = mod.naz;
	c1 = 9.0/8.0;
	c2 = -1.0/24.0;

	// Copy to GPU constant memory
	cudaMemcpyToSymbol(ibndx_d, &ibndx, sizeof(int));
	cudaMemcpyToSymbol(ibndz_d, &ibndz, sizeof(int));
	cudaMemcpyToSymbol(c1_d, &c1, sizeof(int));
	cudaMemcpyToSymbol(c2_d, &c1, sizeof(int));

	wrap_cudaGetLastError("init_cuda_getRecTimes cpyToSymbol");

	// Create timers
	cudaEventCreate(&t0); 
	cudaEventCreate(&t1); 

	// Create streams
	streams = (cudaStream_t*)malloc(nstream*sizeof(cudaStream_t));
	for(int i=0; i<nstream; i++)
    	cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);


	// Tell user we are done
	printfgpu("init_cuda_getRecTimes.");
}//end cuda_init_getRecTimes

void cuda_destroy_getRecTimes(){
/*
Frees previously allocated variables
*/
	// Destroy timers
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);

	// Destroy streams
	int istream;
	for(istream=0; istream<nstream; istream++)
    	cudaStreamDestroy(streams[istream]);
    free(streams);


	// Tell user
	printfgpu("destroy_cuda_getRecTimes.");
}//end cuda_destroy_getRecTimes

// rec_int_p==3 kernels
__global__ void kernel_get_rec_p_3(modPar *d_mod, recPar *d_rec, int isam, float *d_tzz, float *d_rec_p);
__global__ void kernel_get_rec_txx_3(modPar *d_mod, recPar *d_rec, int isam, float *d_txx, float *d_rec_txx);
__global__ void kernel_get_rec_txz_3(modPar *d_mod, recPar *d_rec, int isam, float *d_txz, float *d_rec_txz);
__global__ void kernel_get_rec_pp_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_pp);
__global__ void kernel_get_rec_ss_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_ss);
__global__ void kernel_get_rec_vz_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vz, float *d_rec_vz);
__global__ void kernel_get_rec_vx_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_rec_vx);

// rec_int_p==0,1,2 kernels
__global__ void kernel_get_rec_p_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_tzz, float *d_l2m, float *d_rec_p);
__global__ void kernel_get_rec_tzz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_tzz, float *d_rec_tzz);
__global__ void kernel_get_rec_txx_012(modPar *d_mod, recPar *d_rec, int isam, float *d_txx, float *d_rec_txx);
__global__ void kernel_get_rec_txz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_txz, float *d_rec_txz);
__global__ void kernel_get_rec_pp_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_pp);
__global__ void kernel_get_rec_ss_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_ss);
__global__ void kernel_get_rec_vz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vz, float *d_roz, float *d_tzz, float *d_rec_vz);
__global__ void kernel_get_rec_vx_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_rox, float *d_tzz, float *d_rec_vx);


// Main module function
void cuda_getRecTimes(modPar mod, recPar rec, bndPar bnd, 
				modPar *d_mod, recPar *d_rec, bndPar *d_bnd, 
				int itime, int isam, float *d_vx, float *d_vz, 
				float *d_tzz, float *d_txx, float *d_txz, float *d_l2m, 
				float *d_rox, float *d_roz, float *d_rec_vx, float *d_rec_vz, 
				float *d_rec_txx, float *d_rec_tzz, float *d_rec_txz, float *d_rec_p, 
				float *d_rec_pp, float *d_rec_ss, float *d_rec_udp, float *d_rec_udvz, int verbose)
{
	if (!rec.n) return;

	// Time 
	cudaEventRecord(t0); 

/***********************************************************************
* velocity or txz or potential registrations issues:
* rec_x and rec_z are related to actual txx/tzz/p positions.
* offsets from virtual boundaries must be taken into account.
*
* vx velocities have one sample less in x-direction
* vz velocities have one sample less in z-direction
* txz stresses have one sample less in z-direction and x-direction
*
* Note, in the acoustic scheme P is stored in the Tzz array.
***********************************************************************/

//for (irec=0; irec<rec.n; irec++) { //commented irec loop; GPU version will use nrec threads

		/* interpolation to precise (not necessary on a grid point) position */
		if ( rec.int_p==3 ) {


			/*
			 // Interpolate according to Dirk Kraaijpool's scheme 
			 // Reference:  "Seismic ray fields and ray field maps : theory and algorithms" , 
			 // PhD thesis Utrecht University,Faculty of Geosciences, 2003) 
			 
			 */
			
			if (rec.type.p){
				/* bi-linear interpolation */
				 kernel_get_rec_p_3<<<(rec.n+31)/32, 32,0,streams[0]>>>(d_mod, d_rec, isam, d_tzz, d_rec_p);
			}
			if (rec.type.txx) {
				 kernel_get_rec_txx_3<<<(rec.n+31)/32, 32, 0, streams[1]>>>(d_mod, d_rec, isam, d_txx, d_rec_txx);
			}
			if (rec.type.tzz) {
				 kernel_get_rec_p_3<<<(rec.n+31)/32, 32, 0, streams[2]>>>(d_mod, d_rec, isam, d_tzz, d_rec_tzz);
			}
			if (rec.type.txz) {
				 kernel_get_rec_txz_3<<<(rec.n+31)/32, 32, 0, streams[3]>>>(d_mod, d_rec, isam, d_txz, d_rec_txz);
			}
			if (rec.type.pp) {
				 kernel_get_rec_pp_3<<<(rec.n+31)/32, 32, 0, streams[4]>>>(d_mod, d_rec, isam, d_vx, d_vz, d_rec_pp);
			}
			if (rec.type.ss) {
				 kernel_get_rec_ss_3<<<(rec.n+31)/32, 32, 0, streams[5]>>>(d_mod, d_rec, isam, d_vx, d_vz, d_rec_ss);
			}
			if (rec.type.vz) {
				 kernel_get_rec_vz_3<<<(rec.n+31)/32, 32, 0, streams[6]>>>(d_mod, d_rec, isam, d_vz, d_rec_vz);
			}
			if (rec.type.vx) {
				 kernel_get_rec_vx_3<<<(rec.n+31)/32, 32, 0, streams[7]>>>(d_mod, d_rec, isam, d_vx, d_rec_vx);
			}
		}
		else { /* read values directly from the grid points */
			// if (verbose>=4 && isam==0) {
			 	// printf("cuda_getRecTimes Receiver %d read at gridpoint ix=%d iz=%d",irec, ix, iz);
			// }
			/* interpolation of receivers to same time step is only done for acoustic scheme */
			if (rec.type.p) {
				//kernel_get_rec_p_012<<<>>>();
				kernel_get_rec_p_012<<<(rec.n+31)/32, 32, 0, streams[0]>>>(d_mod, d_rec, isam, d_vx, d_vz, d_tzz, d_l2m, d_rec_p);
			}
			if (rec.type.txx){
				kernel_get_rec_txx_012<<<(rec.n+31)/32, 32, 0, streams[1]>>>(d_mod, d_rec, isam, d_txx, d_rec_txx);
				printf("cuda_getRecTimes WARNING. rec.type.txx needs to be better tested!\n");
			}
			if (rec.type.tzz){
				kernel_get_rec_tzz_012<<<(rec.n+31)/32, 32, 0, streams[2]>>>(d_mod, d_rec, isam, d_tzz, d_rec_tzz);
				printf("cuda_getRecTimes WARNING. rec.type.tzz needs to be better tested!\n");
			}
			if (rec.type.txz) { /* time interpolation to be done */
				kernel_get_rec_txz_012<<<(rec.n+31)/32, 32, 0, streams[3]>>>(d_mod, d_rec, isam, d_txz, d_rec_txz);				
				printf("cuda_getRecTimes WARNING. rec.type.txz needs to be better tested!\n");
			}
			if (rec.type.pp) {
				kernel_get_rec_pp_012<<<(rec.n+31)/32, 32, 0, streams[4]>>>(d_mod, d_rec, isam, d_vx, d_vz, d_rec_pp);
			}
			if (rec.type.ss) {
				kernel_get_rec_ss_012<<<(rec.n+31)/32, 32, 0, streams[5]>>>(d_mod, d_rec, isam, d_vx, d_vz, d_rec_ss);
			}
			if (rec.type.vz) {
				kernel_get_rec_vz_012<<<(rec.n+31)/32, 32, 0, streams[6]>>>(d_mod, d_rec, isam, d_vz, d_roz, d_tzz, d_rec_vz);
			}
			if (rec.type.vx) {
				kernel_get_rec_vx_012<<<(rec.n+31)/32, 32, 0, streams[7]>>>(d_mod, d_rec, isam, d_vx, d_rox, d_tzz, d_rec_vx);
			}
		}

	//} /* end of irec loop */ //commented irec loop; GPU version will use nrec threads

	/* store all x-values on z-level for P Vz for up-down decomposition */
	if (rec.type.ud) {
		printf("cuda_getRecTimes UP/DOWN decomposition not available for GPU implementation.\n");
	}

	wrap_cudaGetLastError("after cuda_getRecTimes");

	cudaDeviceSynchronize();
	cudaEventRecord(t1); 
	cudaEventSynchronize(t1); 
	cudaEventElapsedTime(&tmsec, t0, t1);
	tall += tmsec*1E-3; \
	nrun ++;
}

void cuda_print_getRecTimes_time(){
	printf("cuda_getRecTimes ran %d times and took %.4f s total.\n", nrun, tall);
}

/////////////////////////////////
// int_p == 3 Seismogram kernels
/////////////////////////////////
__global__ void kernel_get_rec_p_3(modPar *d_mod, recPar *d_rec, int isam, float *d_tzz, float *d_rec_p){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get p seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		C00 = d_tzz[ix*n1+iz];
		C10 = d_tzz[(ix+1)*n1+iz];
		C01 = d_tzz[ix*n1+iz+1];
		C11 = d_tzz[(ix+1)*n1+iz+1];
		d_rec_p[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
								  C01*(1.0-rdx)*rdz       + C11*rdx*rdz;	
	}
}// end kernel_get_rec_p_3


__global__ void kernel_get_rec_txx_3(modPar *d_mod, recPar *d_rec, int isam, float *d_txx, float *d_rec_txx){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get txx seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;
	
	if(irec<d_rec->n){
		C00 = d_txx[ix*n1+iz];
		C10 = d_txx[(ix+1)*n1+iz];
		C01 = d_txx[ix*n1+iz+1];
		C11 = d_txx[(ix+1)*n1+iz+1];
		d_rec_txx[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
									C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
	}
}//end kernel_get_rec_txx_3

__global__ void kernel_get_rec_txz_3(modPar *d_mod, recPar *d_rec, int isam, float *d_txz, float *d_rec_txz){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get txz seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		C00 = d_txz[ix2*n1+iz2];
		C10 = d_txz[(ix2+1)*n1+iz2];
		C01 = d_txz[ix2*n1+iz2+1];
		C11 = d_txz[(ix2+1)*n1+iz2+1];
		d_rec_txz[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
											C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
	}
}//end kernel_get_rec_txz_3

__global__ void kernel_get_rec_pp_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_pp){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get pp (div(P)) seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;


	if(irec<d_rec->n){
		C00 = (d_vx[ix2*n1+iz]-d_vx[ix*n1+iz] +
			   d_vz[ix*n1+iz2]-d_vz[ix*n1+iz])/d_mod->dx;
		C10 = (d_vx[(ix2+1)*n1+iz]-d_vx[(ix+1)*n1+iz] +
			   d_vz[(ix+1)*n1+iz2]-d_vz[(ix+1)*n1+iz])/d_mod->dx;
		C01 = (d_vx[ix2*n1+iz+1]-d_vx[ix*n1+iz+1] +
			   d_vz[ix*n1+iz2+1]-d_vz[ix*n1+iz+1])/d_mod->dx;
		C11 = (d_vx[(ix2+1)*n1+iz+1]-d_vx[(ix+1)*n1+iz+1] +
			   d_vz[(ix+1)*n1+iz2+1]-d_vz[(ix+1)*n1+iz+1])/d_mod->dx;
		d_rec_pp[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
								   C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
  	}
}

__global__ void kernel_get_rec_ss_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_ss){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get ss (curl) seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		C00 = (d_vx[ix2*n1+iz2]-d_vx[ix2*n1+iz] -
			   (d_vz[ix2*n1+iz2]-d_vz[ix*n1+iz2]))/d_mod->dx;
		C10 = (d_vx[(ix2+1)*n1+iz2]-d_vx[(ix2+1)*n1+iz] -
				(d_vz[(ix2+1)*n1+iz2]-d_vz[(ix+1)*n1+iz2]))/d_mod->dx;
		C01 = (d_vx[ix2*n1+iz2+1]-d_vx[ix2*n1+iz+1] -
				(d_vz[ix2*n1+iz2+1]-d_vz[ix*n1+iz2+1]))/d_mod->dx;;
		C11 = (d_vx[(ix2+1)*n1+iz2+1]-d_vx[(ix2+1)*n1+iz+1] -
				(d_vz[(ix2+1)*n1+iz2+1]-d_vz[(ix+1)*n1+iz2+1]))/d_mod->dx;
		d_rec_ss[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
								   C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
	}
}

__global__ void kernel_get_rec_vz_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vz, float *d_rec_vz){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get vz seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		C00 = d_vz[ix*n1+iz2];
		C10 = d_vz[(ix+1)*n1+iz2];
		C01 = d_vz[ix*n1+iz2+1];
		C11 = d_vz[(ix+1)*n1+iz2+1];
		d_rec_vz[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
								   C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
	}
}

__global__ void kernel_get_rec_vx_3(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_rec_vx){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get vx seismogram, interpolation type 3
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float rdz, rdx, C00, C10, C01, C11;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = (int)floorf(d_rec->zr[irec]/d_mod->dz)+ibndz_d;
	ix = (int)floorf(d_rec->xr[irec]/d_mod->dx)+ibndx_d;
	rdz = (d_rec->zr[irec] - (iz-ibndz_d)*d_mod->dz)/d_mod->dz;
	rdx = (d_rec->xr[irec] - (ix-ibndx_d)*d_mod->dx)/d_mod->dx;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		C00 = d_vx[ix2*n1+iz];
		C10 = d_vx[(ix2+1)*n1+iz];
		C01 = d_vx[ix2*n1+iz+1];
		C11 = d_vx[(ix2+1)*n1+iz+1];
		d_rec_vx[irec*ntrec+isam] = C00*(1.0-rdx)*(1.0-rdz) + C10*rdx*(1.0-rdz) +
								   C01*(1.0-rdx)*rdz       + C11*rdx*rdz;
	}
}

/////////////////////////////////////
// end int_p == 3 Seismogram kernels
/////////////////////////////////////

/////////////////////////////////////
// int_p == 0,1,2 Seismogram kernels
/////////////////////////////////////
__global__ void kernel_get_rec_p_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_tzz, float *d_l2m, float *d_rec_p){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get p seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float dvx, dvz;
	float field = 0;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		if (d_rec->int_p == 1) {
			if (d_mod->ischeme == 1) { /* interpolate Tzz times -1/2 Dt backward to Vz times */
                dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                      c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
                dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                      c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
                field = d_tzz[ix*n1+iz] + 0.5*d_l2m[ix*n1+iz]*(dvx+dvz);
                dvx = c1_d*(d_vx[(ix+1)*n1+iz1] - d_vx[ix*n1+iz1]) +
                      c2_d*(d_vx[(ix+2)*n1+iz1] - d_vx[(ix-1)*n1+iz1]);
                dvz = c1_d*(d_vz[ix*n1+iz1+1]   - d_vz[ix*n1+iz1]) +
                      c2_d*(d_vz[ix*n1+iz1+2]   - d_vz[ix*n1+iz1-1]);
                field += d_tzz[ix*n1+iz1] + 0.5*d_l2m[ix*n1+iz1]*(dvx+dvz);
				d_rec_p[irec*ntrec+isam] = 0.5*field;
			}
			else {
				d_rec_p[irec*ntrec+isam] = 0.5*(d_tzz[ix*n1+iz1]+d_tzz[ix*n1+iz]);
			}
		}
		else if (d_rec->int_p == 2) {
			if (d_mod->ischeme == 1) { /* interpolate Tzz times -1/2 Dt backward to Vx times */
                dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                      c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
                dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                      c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
                field = d_tzz[ix*n1+iz] + 0.5*d_l2m[ix*n1+iz]*(dvx+dvz);
                dvx = c1_d*(d_vx[(ix1+1)*n1+iz] - d_vx[ix1*n1+iz]) +
                      c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix1-1)*n1+iz]);
                dvz = c1_d*(d_vz[ix1*n1+iz+1]   - d_vz[ix1*n1+iz]) +
                      c2_d*(d_vz[ix1*n1+iz+2]   - d_vz[ix1*n1+iz-1]);
                field += d_tzz[ix1*n1+iz] + 0.5*d_l2m[ix1*n1+iz]*(dvx+dvz);
				d_rec_p[irec*ntrec+isam] = 0.5*field;
			}
			else {
				d_rec_p[irec*ntrec+isam] = 0.5*(d_tzz[ix1*n1+iz]+d_tzz[ix*n1+iz]);
			}
		}
		else {
			d_rec_p[irec*ntrec+isam] = d_tzz[ix*n1+iz];
		}
	}
}

__global__ void kernel_get_rec_txx_012(modPar *d_mod, recPar *d_rec, int isam, float *d_txx, float *d_rec_txx){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get txx seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		d_rec_txx[irec*ntrec+isam] = d_txx[ix*n1+iz];				
	}
}

__global__ void kernel_get_rec_tzz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_tzz, float *d_rec_tzz){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get tzz seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		d_rec_tzz[irec*ntrec+isam] = d_tzz[ix*n1+iz];				
	}
}


__global__ void kernel_get_rec_txz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_txz, float *d_rec_txz){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get txz seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		if (d_rec->int_vz == 2 || d_rec->int_vx == 2) {
					d_rec_txz[irec*ntrec+isam] = 0.25*(
							d_txz[ix*n1+iz2]+d_txz[ix2*n1+iz2]+
							d_txz[ix*n1+iz]+d_txz[ix2*n1+iz]);
				}
				else {
					d_rec_txz[irec*ntrec+isam] = d_txz[ix2*n1+iz2];
				}
	}
}

__global__ void kernel_get_rec_pp_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_pp){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get pp (div) seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float dvx, dvz;
	float field = 0;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		d_rec_pp[irec*ntrec+isam] = (d_vx[ix2*n1+iz]-d_vx[ix*n1+iz] +
											d_vz[ix*n1+iz2]-d_vz[ix*n1+iz])/d_mod->dx;

	}
}

__global__ void kernel_get_rec_ss_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_vz, float *d_rec_ss){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get ss (rot) seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float dvx, dvz;
	float field = 0;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
		d_rec_ss[irec*ntrec+isam] = (d_vx[ix2*n1+iz2]-d_vx[ix2*n1+iz] -
										   (d_vz[ix2*n1+iz2]-d_vz[ix*n1+iz2]))/d_mod->dx;

	}
}

__global__ void kernel_get_rec_vz_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vz, float *d_roz, float *d_tzz, float *d_rec_vz){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get p seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float dvx, dvz;
	float field = 0;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
/* interpolate vz to vx position to the right and above of vz */
		if (d_rec->int_vz == 1) {
			d_rec_vz[irec*ntrec+isam] = 0.25*(
					d_vz[ix*n1+iz2]+d_vz[ix1*n1+iz2]+
					d_vz[ix*n1+iz] +d_vz[ix1*n1+iz]);
		}
/* interpolate vz to Txx/Tzz position by taking the mean of 2 values */
		else if (d_rec->int_vz == 2) {
			if (d_mod->ischeme == 1) { /* interpolate Vz times +1/2 Dt forward to P times */
                field = d_vz[ix*n1+iz] - 0.5*d_roz[ix*n1+iz]*(
                	c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
                	c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
                field += d_vz[ix*n1+iz2] - 0.5*d_roz[ix*n1+iz2]*(
                	c1_d*(d_tzz[ix*n1+iz2]   - d_tzz[ix*n1+iz2-1]) +
                	c2_d*(d_tzz[ix*n1+iz2+1] - d_tzz[ix*n1+iz2-2]));
				d_rec_vz[irec*ntrec+isam] = 0.5*field;
			}
			else {
				d_rec_vz[irec*ntrec+isam] = 0.5*(d_vz[ix*n1+iz2]+d_vz[ix*n1+iz]);
			}
		}
		else {
			d_rec_vz[irec*ntrec+isam] = d_vz[ix*n1+iz2];
			//rec_vz[irec*rec.nt+isam] = vz[ix*n1+iz];
			//fprintf(stderr,"isam=%d vz[%d]=%e vz[%d]=%e vz[%d]=%e \n",isam, iz-1,vz[ix*n1+iz-1],iz,vz[ix*n1+iz], iz+1, vz[ix*n1+iz+1]);
		}
	}
}

__global__ void kernel_get_rec_vx_012(modPar *d_mod, recPar *d_rec, int isam, float *d_vx, float *d_rox, float *d_tzz, float *d_rec_vx){
/*
Call: <<<(d_rec.n+255)/256, 256>>>
Optimizations: probably shmem usage

Get p seismogram, interpolation types 0,1,2
*/
	int irec = threadIdx.x + blockDim.x*blockIdx.x;

	int iz, ix, iz1, ix1, iz2, ix2;
	float dvx, dvz;
	float field = 0;

	int n1 = d_mod->naz;
	int ntrec = d_rec->nt;

	iz = d_rec->z[irec]+ibndz_d;
	ix = d_rec->x[irec]+ibndx_d;
	iz1 = iz-1;
	ix1 = ix-1;
	iz2 = iz+1;
	ix2 = ix+1;

	if(irec<d_rec->n){
/* interpolate vx to vz position to the left and below of vx */
		if (d_rec->int_vx == 1) {
			d_rec_vx[irec*ntrec+isam] = 0.25*(
					d_vx[ix2*n1+iz]+d_vx[ix2*n1+iz1]+
					d_vx[ix*n1+iz]+d_vx[ix*n1+iz1]);
		}
/* interpolate vx to Txx/Tzz position by taking the mean of 2 values */
		else if (d_rec->int_vx == 2) {
			if (d_mod->ischeme == 1) { /* interpolate Vx times +1/2 Dt forward to P times */
    			field = d_vx[ix*n1+iz] - 0.5*d_rox[ix*n1+iz]*(
        			c1_d*(d_tzz[ix*n1+iz]     - d_tzz[(ix-1)*n1+iz]) +
        			c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
    			field += d_vx[ix2*n1+iz] - 0.5*d_rox[ix2*n1+iz]*(
        			c1_d*(d_tzz[ix2*n1+iz]     - d_tzz[(ix2-1)*n1+iz]) +
        			c2_d*(d_tzz[(ix2+1)*n1+iz] - d_tzz[(ix2-2)*n1+iz]));
				d_rec_vx[irec*ntrec+isam] = 0.5*field;
			}
			else {
				d_rec_vx[irec*ntrec+isam] = 0.5*(d_vx[ix2*n1+iz]+d_vx[ix*n1+iz]);
			}
		}
		else {
			d_rec_vx[irec*ntrec+isam] = d_vx[ix2*n1+iz];
		}
	}
}

/////////////////////////////////////
// end int_p == 0,1,2 Seismogram kernels
/////////////////////////////////////