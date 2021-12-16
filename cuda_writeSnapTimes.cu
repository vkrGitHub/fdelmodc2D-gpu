extern "C"{
	#include <assert.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <errno.h>
	#include <math.h>
	#include <string.h>
	#include "par.h"
	#include "segy.h"
	#include "fdelmodc.h"		
}

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "cuda_myutils.cuh"
#include "cuda_fileOpen.cuh"	

#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#define ISODD(n) ((n) & 01)
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
/**
*  Writes gridded wavefield(s) at a desired time to output file(s) 
*
*   AUTHOR:
*           Jan Thorbecke (janth@xs4all.nl)
*           The Netherlands 
**/

/**
* GPU version - Victor Koehne, SENAI CIMATEC, Brazil
* 
* Optimizations:
* - Use shmem on each kernel
* - Use streams to do concurrently the kernels for every .type of snap
*
* Problems:
* - Initializing (and destroying) *d_snp from main function is giving errors at the 
* cudaMemcpy D2H that comes after each resampling kernel; it throws 'invalid error (code1)'; 
* for this reason init is called at the first call of the snap saving function and free is called in 
* the last call.
**/


static float *d_snp = NULL;

void cuda_init_writeSnapTimes(snaPar sna){
/*
Initializes snap mtx 
*/
	cudaMalloc(&d_snp, sna.nz*sizeof(float));

	wrap_cudaGetLastError("cuda_init_writeSnapTimes err alloc");

	printfgpu("cuda_init_writeSnapTimes");
}

void cuda_destroy_writeSnapTimes(){
/*
Frees snap mtx 
*/
	cudaFree(d_snp);

	wrap_cudaGetLastError("cuda_destroy_writeSnapTimes err free");
	printfgpu("cuda_destroy_writeSnapTimes");
}

__global__ void kernel_getSnapZline(int izs, int ize, int skipdz, int izshift, int ix2, int n1, int n1r, float *d_field, float *d_resamp){
	/*
	Call: <<<(sna.nz+255)/256,256>>>
	Gets z-stripe, resampled from a field into d_snap
	*/
	
		int j = blockDim.x*blockIdx.x + threadIdx.x;
		int iz = izs + j*skipdz;
	
		if (j<n1r){
			d_resamp[j] = d_field[ix2*n1+iz+izshift];
		}
	
}

__global__ void kernel_getSnapZline_div(int izs, int ize, int skipdz, int sdx, int ix, int n1, int n1r, float *d_vx, float *d_vz, float *d_resamp){
/*
Call: <<<(sna.nz+255)/256,256>>>
Gets z-stripe, resampled from a field into d_snap
*/

	int j = blockDim.x*blockIdx.x + threadIdx.x;
	int iz = izs + j*skipdz;

	if (j<n1r){
		d_resamp[j] = sdx*((d_vx[(ix+1)*n1+iz]-d_vx[ix*n1+iz])+(d_vz[ix*n1+iz+1]-d_vz[ix*n1+iz]));
	}

}

__global__ void kernel_getSnapZline_rot(int izs, int ize, int skipdz, int sdx, int ix, int n1, int n1r, float *d_vx, float *d_vz, float *d_resamp){
/*
Call: <<<(sna.nz+255)/256,256>>>
Gets z-stripe, resampled from a field into d_snap
*/

	int j = blockDim.x*blockIdx.x + threadIdx.x;
	int iz = izs + j*skipdz;

	if (j<n1r){
		d_resamp[j] = sdx*((d_vx[ix*n1+iz]-d_vx[ix*n1+iz-1])-
									(d_vz[ix*n1+iz]-d_vz[(ix-1)*n1+iz]));
	}

}






void cuda_writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav, 
						int ixsrc, int izsrc, int itime, float *d_vx, float *d_vz, 
						float *d_tzz, float *d_txx, float *d_txz, int verbose)
{
	FILE    *fpvx, *fpvz, *fptxx, *fptzz, *fptxz, *fpp, *fppp, *fpss;
	int append, isnap;
	static int first=1, last=0;
	int n1, ibndx, ibndz, ixs, izs, ize, i, j;
	int ix, iz, ix2;
	float *snap, sdx, stime;
	segy hdr;

	if (sna.nsnap==0) return;

	char gpufname[400];
    strcpy(gpufname, sna.file_snap);
    name_ext(gpufname, "-gpu");

    ibndx = mod.ioXx;
    ibndz = mod.ioXz;
	n1    = mod.naz;
	sdx   = 1.0/mod.dx;

	if (sna.withbnd) {
		sna.nz=mod.naz;
		sna.z1=0;
		sna.z2=mod.naz-1;
		sna.skipdz=1;

		sna.nx=mod.nax;
		sna.x1=0;
		sna.x2=mod.nax-1;
		sna.skipdx=1;
	}

	/* check if this itime is a desired snapshot time */
	if ( (((itime-sna.delay) % sna.skipdt)==0) && 
		  (itime >= sna.delay) &&
		  (itime <= sna.delay+(sna.nsnap-1)*sna.skipdt) ) {

		isnap = NINT((itime-sna.delay)/sna.skipdt);

        if (mod.grid_dir) stime = (-wav.nt+1+itime+1)*mod.dt;  /* reverse time modeling */
        else  stime = itime*mod.dt;
		if (verbose) vmess("Writing snapshot(%d) at time=%.4f", isnap+1, stime);
	
		if (first) {
			append=0;
			first=0;
		}
		else {
			append=1;
		}

		if(itime==sna.delay) cuda_init_writeSnapTimes(sna);


		if (sna.type.vx)  fpvx  = fileOpen(gpufname, "_svx", append);
		if (sna.type.vz)  fpvz  = fileOpen(gpufname, "_svz", append);
		if (sna.type.p)   fpp   = fileOpen(gpufname, "_sp", append);
		if (sna.type.txx) fptxx = fileOpen(gpufname, "_stxx", append);
		if (sna.type.tzz) fptzz = fileOpen(gpufname, "_stzz", append);
		if (sna.type.txz) fptxz = fileOpen(gpufname, "_stxz", append);
		if (sna.type.pp)  fppp  = fileOpen(gpufname, "_spp", append);
		if (sna.type.ss)  fpss  = fileOpen(gpufname, "_sss", append);

		memset(&hdr,0,TRCBYTES);
		hdr.dt     = 1000000*(sna.skipdt*mod.dt);
		hdr.ungpow  = (sna.delay*mod.dt);
		hdr.scalco = -1000;
		hdr.scalel = -1000;
		hdr.sx     = 1000*(mod.x0+ixsrc*mod.dx);
		hdr.sdepth = 1000*(mod.z0+izsrc*mod.dz);
		hdr.fldr   = isnap+1;
		hdr.trid   = 1;
		hdr.ns     = sna.nz;
		hdr.trwf   = sna.nx;
		hdr.ntr    = (isnap+1)*sna.nx;
		hdr.f1     = sna.z1*mod.dz+mod.z0;
		hdr.f2     = sna.x1*mod.dx+mod.x0;
		hdr.d1     = mod.dz*sna.skipdz;
		hdr.d2     = mod.dx*sna.skipdx;
		if (sna.withbnd) {
        	if ( !ISODD(bnd.top)) hdr.f1 = mod.z0 - bnd.ntap*mod.dz;
        	if ( !ISODD(bnd.lef)) hdr.f2 = mod.x0 - bnd.ntap*mod.dx;
        	//if ( !ISODD(bnd.rig)) ;
        	//if ( !ISODD(bnd.bot)) store=1;
		}

/***********************************************************************
* vx velocities have one sample less in x-direction
* vz velocities have one sample less in z-direction
* txz stresses have one sample less in z-direction and x-direction
***********************************************************************/

		snap = (float *)malloc(sna.nz*sizeof(float));

		/* Decimate, with skipdx and skipdz, the number of gridpoints written to file 
		   and write to file. */
		for (ixs=sna.x1, i=0; ixs<=sna.x2; ixs+=sna.skipdx, i++) {
			hdr.tracf  = i+1;
			hdr.tracl  = isnap*sna.nx+i+1;
			hdr.gx     = 1000*(mod.x0+ixs*mod.dx);
			ix = ixs+ibndx;
			ix2 = ix+1;

			izs = sna.z1+ibndz;
			ize = sna.z2+ibndz;

			if (sna.withbnd) {
				izs = 0;
				ize = sna.z2;
				ix = ixs;
				ix2 = ix;
				if (sna.type.vz || sna.type.txz) izs = -1;
        		if ( !ISODD(bnd.lef)) hdr.gx = 1000*(mod.x0 - bnd.ntap*mod.dx);
			}

			if (sna.type.vx) {
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 0, ix2, n1, sna.nz, d_vx, d_snp);
				
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fpvx);
			}
			if (sna.type.vz) { 
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 1, ix, n1, sna.nz, d_vz, d_snp);
				
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fpvz);

			}
			if (sna.type.p) {
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 0, ix, n1, sna.nz, d_tzz, d_snp);
				
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fpp);
			}
			if (sna.type.tzz) {
				//kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, ix, n1, sna.nz, d_tzz, d_snp);
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 0, ix, n1, sna.nz, d_tzz, d_snp);
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fptzz);
				vmess("cuda_writeSnapTimes (snap tzz) needs to be tested more thoroughly!");
			}
			if (sna.type.txx) {
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 0, ix, n1, sna.nz, d_txx, d_snp);
				
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fptxx);
				vmess("cuda_writeSnapTimes (snap txx) needs to be tested more thoroughly!");

			}
			if (sna.type.txz) {
				kernel_getSnapZline<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, 1, ix2, n1, sna.nz, d_txz, d_snp);
				
				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fptxz);
				vmess("cuda_writeSnapTimes (snap txz) needs to be tested more thoroughly!");

			}
			/* calculate divergence of velocity field */
			if (sna.type.pp) {
				kernel_getSnapZline_div<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, sdx, ix, n1, sna.nz, d_vx, d_vz, d_snp);

				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fppp);
			}
			/* calculate rotation of velocity field */
			if (sna.type.ss) {
				kernel_getSnapZline_rot<<<(sna.nz+255)/256,256>>>(izs, ize, sna.skipdz, sdx, ix, n1, sna.nz, d_vx, d_vz, d_snp);

				cuda_traceWrite(&hdr, d_snp, sna.nz, (cudaStream_t) 0, fpss);
			}

		}

		if(itime == sna.delay+(sna.nsnap-1)*sna.skipdt) cuda_destroy_writeSnapTimes();

		if (sna.type.vx) fclose(fpvx);
		if (sna.type.vz) fclose(fpvz);
		if (sna.type.p) fclose(fpp);
		if (sna.type.txx) fclose(fptxx);
		if (sna.type.tzz) fclose(fptzz);
		if (sna.type.txz) fclose(fptxz);
		if (sna.type.pp) fclose(fppp);
		if (sna.type.ss) fclose(fpss);

		free(snap);
	}

	wrap_cudaGetLastError("cuda_writeSnapTimes");
	//	return 0;
}

