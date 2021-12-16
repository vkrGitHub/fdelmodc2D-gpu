#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include"fdelmodc.h"

#include "cuda_myutils.cuh"

#define ISODD(n) ((n) & 01)

static float *d_src1x, *d_src1z, *d_src2x, *d_src2z;

// Variables below will store addresses for module variables above
// This way, they can be accessed by the GPU without having to 
// pass variable's adresses as kernel arguments
__constant__ float *dd_src1x, *dd_src1z, *dd_src2x, *dd_src2z;
//static int first=1;

void cuda_allocStoreSourceOnSurface(srcPar src, int verbose)
{
    /* allocated 2x size for dipole Potential sources */
    cudaMalloc(&d_src1x, 2*src.n*sizeof(float));
	cudaMalloc(&d_src1z, 2*src.n*sizeof(float));
	cudaMalloc(&d_src2x, 2*src.n*sizeof(float));
	cudaMalloc(&d_src2z, 2*src.n*sizeof(float));

    cudaMemcpyToSymbol(dd_src1x, &d_src1x, 1*sizeof(float*));
    cudaMemcpyToSymbol(dd_src1z, &d_src1z, 1*sizeof(float*));
    cudaMemcpyToSymbol(dd_src2x, &d_src2x, 1*sizeof(float*));
    cudaMemcpyToSymbol(dd_src2z, &d_src2z, 1*sizeof(float*));
	
	cudaMemset(d_src1x, 0, 2*src.n*sizeof(float));
	cudaMemset(d_src1z, 0, 2*src.n*sizeof(float));
	cudaMemset(d_src2x, 0, 2*src.n*sizeof(float));
	cudaMemset(d_src2z, 0, 2*src.n*sizeof(float));

    if(verbose>=1) printfgpu("cuda_allocStoreSourceOnSurface.");

    wrap_cudaGetLastError("cuda_allocStoreSourceOnSurface.");
}

void cuda_freeStoreSourceOnSurface(int verbose)
{
    cudaFree(d_src1x);
    cudaFree(d_src1z);
    cudaFree(d_src2x);
    cudaFree(d_src2z);

    if(verbose>=1) printfgpu("cuda_freeStoreSourceOnSurface.");

    wrap_cudaGetLastError("cuda_freeStoreSourceOnSurface.");   
}

__global__ void kernel_storeSourceOnSurface(modPar *d_mod, srcPar *d_src, bndPar *d_bnd, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, int verbose)
{
/**********************************************************************
Call:<<<1,1>>>

   AUTHOR:
           Jan Thorbecke (janth@xs4all.nl)
           The Netherlands 

           GPU mod -  Victor Koehne
           Senai-Cimatec, Brazil
***********************************************************************/

	int   ixs, izs, isrc, is0;
    int   ibndz, ibndx, store;
	int   nx, nz, n1;

	nx  = d_mod->nx;
	nz  = d_mod->nz;
	n1  = d_mod->naz;

	if (d_src->type==6) {
    	ibndz = d_mod->ioXz;
    	ibndx = d_mod->ioXx;
	}
	else if (d_src->type==7) {
    	ibndz = d_mod->ioZz;
    	ibndx = d_mod->ioZx;
	}
	else if (d_src->type==2) {
    	ibndz = d_mod->ioTz;
    	ibndx = d_mod->ioTx;
    	if (d_bnd->lef==4 || d_bnd->lef==2) ibndx += d_bnd->ntap;
	}
	else {	
    	ibndz = d_mod->ioPz;
    	ibndx = d_mod->ioPx;
    	if (d_bnd->lef==4 || d_bnd->lef==2) ibndx += d_bnd->ntap;
	}

/* check if there are sources placed on the boundaries */
	//is0 = -1*floor((d_src->n-1)/2);
    is0 = -1*floorf((d_src->n-1)/2);
#pragma omp	for private (isrc, ixs, izs, store) 
	for (isrc=0; isrc<(d_src->n); isrc++) {
		/* calculate the source position */
		if (d_src->random || d_src->multiwav) {
			ixs = d_src->x[isrc] + ibndx;
			izs = d_src->z[isrc] + ibndz;
		}
		else { /* plane wave and point sources */
			ixs = ixsrc + ibndx + is0 + isrc;
			izs = izsrc + ibndz;
		}

//        printf("source at x=%d bounds at %d %d : %d %d ", ixs, ibndx+1, nx+ibndx, d_mod->ioXz, d_mod->ieXz);
//        printf("source at z=%d bounds at %d %d : %d %d ", izs, ibndz+1, nz+ibndz, d_mod->ioXx, d_mod->ieXx);

/* check if there are source positions placed on the boundaries. 
 * In that case store them and reapply them after the boundary
 * conditions have been set */

        store=0;
		if ( (ixs <= ibndx+1)  && ISODD(d_bnd->lef)) store=1;
		if ( (ixs >= nx+ibndx) && ISODD(d_bnd->rig)) store=1;
		if ( (izs <= ibndz+1)  && ISODD(d_bnd->top)) store=1;
		if ( (izs >= nz+ibndz) && ISODD(d_bnd->bot)) store=1;

		if (d_mod->ischeme <= 2) { /* Acoustic scheme */
            
            if (store) {
                if (verbose>=5) printf("cuda_sto...ace source at x=%d z=%d stored before free surface\n", ixs, izs);

                /* Compressional source */
                if (d_src->type == 1) {
                
                    if (d_src->orient==1) { /* monopole */
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                    }
                    else if (d_src->orient==2) { /* dipole +/- */
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2z[isrc] = d_tzz[ixs*n1+izs+1];
                    }
                    else if (d_src->orient==3) { /* dipole - + */
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2z[isrc] = d_tzz[(ixs-1)*n1+izs];
                    }
                    else if (d_src->orient==4) { /* dipole +/0/- */
                        if (izs > ibndz) 
                            dd_src1z[isrc] = d_tzz[ixs*n1+izs-1];
                        if (izs < d_mod->nz+ibndz-1) 
                            dd_src2z[isrc] = d_tzz[ixs*n1+izs+1];
                    }
                    else if (d_src->orient==5) { /* dipole + - */
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2z[isrc] = d_tzz[(ixs+1)*n1+izs];
                    }
                }
                else if (d_src->type==6) {
                    dd_src1x[isrc] = d_vx[ixs*n1+izs];
                }
                else if (d_src->type==7) {
                    dd_src1z[isrc] = d_vz[ixs*n1+izs];
                }
                
            }
        }
        else { /* Elastic scheme */

          	if (store) {
                if (verbose>=5) printf("cuda_sto...ace source at x=%d z=%d stored before free surface\n", ixs, izs);

              	if (d_src->type==1) {
                    if (d_src->orient==1) { /* monopole */
                        dd_src1x[isrc] = d_txx[ixs*n1+izs];
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                    }
                    else if (d_src->orient==2) { /* dipole +/- */
                        dd_src1x[isrc] = d_txx[ixs*n1+izs];
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2x[isrc] = d_txx[ixs*n1+izs+1];
                        dd_src2z[isrc] = d_tzz[ixs*n1+izs+1];
                    }
                    else if (d_src->orient==3) { /* dipole - + */
                        dd_src1x[isrc] = d_txx[ixs*n1+izs];
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2x[isrc] = d_txx[(ixs-1)*n1+izs];
                        dd_src2z[isrc] = d_tzz[(ixs-1)*n1+izs];
                    }
                    else if (d_src->orient==4) {  /* dipole +/0/- */
                        if (izs > ibndz) {
                            dd_src1x[isrc] = d_txx[ixs*n1+izs-1];
                            dd_src1z[isrc] = d_tzz[ixs*n1+izs-1];
                        }
                        if (izs < d_mod->nz+ibndz-1) {
                            dd_src1x[isrc] = d_txx[ixs*n1+izs+1];
                            dd_src1z[isrc] = d_tzz[ixs*n1+izs+1];
                        }
                    }
                    else if (d_src->orient==5) { /* dipole + - */
                        dd_src1x[isrc] = d_txx[ixs*n1+izs];
                        dd_src1z[isrc] = d_tzz[ixs*n1+izs];
                        dd_src2x[isrc] = d_txx[(ixs+1)*n1+izs];
                        dd_src2z[isrc] = d_tzz[(ixs+1)*n1+izs];
                    }
              	}
              	else if (d_src->type==2) {
                    
                    /* d_txz source */
                    if ((izs == ibndz) && d_bnd->top==1) {
                        dd_src1x[isrc] = d_txz[(ixs-1)*n1+izs-1];
                        dd_src2x[isrc] = d_txz[ixs*n1+izs-1];
                    }
                    else {
                        dd_src1x[isrc] = d_txz[ixs*n1+izs];
                    }
                    /* possible dipole orientations for a d_txz source */
                    if (d_src->orient == 2) { /* dipole +/- */
                        dd_src2x[isrc] = d_txz[ixs*n1+izs+1];
                    }
                    else if (d_src->orient == 3) { /* dipole - + */
                        dd_src2x[isrc] = d_txz[(ixs-1)*n1+izs];
                    }
                    else if (d_src->orient == 4) { /*  dipole +/O/- */
                        /* correction: subtrace previous value to prevent z-1 values. */
                        dd_src1x[isrc] = d_txz[ixs*n1+izs];
                        dd_src2x[isrc] = d_txz[ixs*n1+izs+1];
                    }
                    else if (d_src->orient == 5) { /* dipole + - */
                        dd_src2x[isrc] = d_txz[(ixs+1)*n1+izs];
                    }

              	}
               	else if (d_src->type==3) {
                   	dd_src1z[isrc] = d_tzz[ixs*n1+izs];
               	}
               	else if (d_src->type==4) {
                   	dd_src1x[isrc] = d_txx[ixs*n1+izs];
               	}
               	else if (d_src->type==5) {
                                        
                    dd_src1x[isrc] = d_vx[ixs*n1+izs];
                    dd_src1z[isrc] = d_vz[ixs*n1+izs];
                    dd_src2x[isrc] = d_vx[ixs*n1+izs-1];
                    dd_src2z[isrc] = d_vz[(ixs-1)*n1+izs];

                    /* determine second position of dipole */
                    if (d_src->orient == 2) { /* dipole +/- vertical */
                        izs += 1;
                        dd_src1x[isrc+d_src->n] = d_vx[ixs*n1+izs];
                        dd_src1z[isrc+d_src->n] = d_vz[ixs*n1+izs];
                        dd_src2x[isrc+d_src->n] = d_vx[ixs*n1+izs-1];
                        dd_src2z[isrc+d_src->n] = d_vz[(ixs-1)*n1+izs];
                    }
                    else if (d_src->orient == 3) { /* dipole - + horizontal */
                        ixs += 1;
                        dd_src1x[isrc+d_src->n] = d_vx[ixs*n1+izs];
                        dd_src1z[isrc+d_src->n] = d_vz[ixs*n1+izs];
                        dd_src2x[isrc+d_src->n] = d_vx[ixs*n1+izs-1];
                        dd_src2z[isrc+d_src->n] = d_vz[(ixs-1)*n1+izs];
                    }

				}
               	else if (d_src->type==6) {
                   	dd_src1x[isrc] = d_vx[ixs*n1+izs];
               	}
               	else if (d_src->type==7) {
                   	dd_src1z[isrc] = d_vz[ixs*n1+izs];
               	}
               	else if (d_src->type==8) {
                    
                    dd_src1x[isrc] = d_vx[(ixs+1)*n1+izs];
                    dd_src1z[isrc] = d_vz[ixs*n1+izs+1];
                    dd_src2x[isrc] = d_vx[ixs*n1+izs];
                    dd_src2z[isrc] = d_vz[ixs*n1+izs];
                    
                    /* determine second position of dipole */
                    if (d_src->orient == 2) { /* dipole +/- vertical */
                        izs += 1;
                        dd_src1x[isrc+d_src->n] = d_vx[(ixs+1)*n1+izs];
                        dd_src1z[isrc+d_src->n] = d_vz[ixs*n1+izs+1];
                        dd_src2x[isrc+d_src->n] = d_vx[ixs*n1+izs];
                        dd_src2z[isrc+d_src->n] = d_vz[ixs*n1+izs];
                    }
                    else if (d_src->orient == 3) { /* dipole - + horizontal */
                        ixs += 1;
                        dd_src1x[isrc+d_src->n] = d_vx[(ixs+1)*n1+izs];
                        dd_src1z[isrc+d_src->n] = d_vz[ixs*n1+izs+1];
                        dd_src2x[isrc+d_src->n] = d_vx[ixs*n1+izs];
                        dd_src2z[isrc+d_src->n] = d_vz[ixs*n1+izs];
                    }

               	} /* end of source.type */
           	}
		}
    }
    
}

    
    
__global__  void kernel_reStoreSourceOnSurface(modPar *d_mod, srcPar *d_src, bndPar *d_bnd, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, int verbose)
{
/**********************************************************************
Call:<<<1,1>>>

   AUTHOR:
           Jan Thorbecke (janth@xs4all.nl)
           The Netherlands 

           GPU mod -  Victor Koehne
            Senai-Cimatec, Brazil
***********************************************************************/
    
	int   ixs, izs, isrc, is0;
    int   ibndz, ibndx, store;
	int   nx, nz, n1;
    
	nx  = d_mod->nx;
	nz  = d_mod->nz;
	n1  = d_mod->naz;
    
	if (d_src->type==6) {
    	ibndz = d_mod->ioXz;
    	ibndx = d_mod->ioXx;
	}
	else if (d_src->type==7) {
    	ibndz = d_mod->ioZz;
    	ibndx = d_mod->ioZx;
	}
	else if (d_src->type==2) {
    	ibndz = d_mod->ioTz;
    	ibndx = d_mod->ioTx;
    	if (d_bnd->lef==4 || d_bnd->lef==2) ibndx += d_bnd->ntap;
	}
	else {	
    	ibndz = d_mod->ioPz;
    	ibndx = d_mod->ioPx;
    	if (d_bnd->lef==4 || d_bnd->lef==2) ibndx += d_bnd->ntap;
	}

	/* restore source positions on the edge */
	//is0 = -1*floor((d_src->n-1)/2);
    is0 = -1*floorf((d_src->n-1)/2);
#pragma omp	for private (isrc, ixs, izs, store) 
	for (isrc=0; isrc<d_src->n; isrc++) {
		/* calculate the source position */
		if (d_src->random || d_src->multiwav) {
			ixs = d_src->x[isrc] + ibndx;
			izs = d_src->z[isrc] + ibndz;
		}
		else { /* plane wave and point sources */
			ixs = ixsrc + ibndx + is0 + isrc;
			izs = izsrc + ibndz;
		}
        
        store=0;
		if ( (ixs <= ibndx+1)  && ISODD(d_bnd->lef)) store=1;
		if ( (ixs >= nx+ibndx) && ISODD(d_bnd->rig)) store=1;
		if ( (izs <= ibndz+1)  && ISODD(d_bnd->top)) store=1;
		if ( (izs >= nz+ibndz) && ISODD(d_bnd->bot)) store=1;
        
		if (d_mod->ischeme <= 2) { /* Acoustic scheme */
            
            if (store) {
                if (verbose>=5) printf("cuda_reSt...ace source at x=%d z=%d restored at free surface\n", ixs, izs);

                /* Compressional source */
                if (d_src->type == 1) {
                    
                    if (d_src->orient==1) { /* monopole */
                        d_tzz[ixs*n1+izs]= dd_src1z[isrc];
                    }
                    else if (d_src->orient==2) { /* dipole +/- */
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_tzz[ixs*n1+izs+1] = dd_src2z[isrc];
                    }
                    else if (d_src->orient==3) { /* dipole - + */
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_tzz[(ixs-1)*n1+izs] = dd_src2z[isrc];
                    }
                    else if (d_src->orient==4) { /* dipole +/0/- */
                        if (izs > ibndz) 
                            d_tzz[ixs*n1+izs-1] = dd_src1z[isrc];
                        if (izs < d_mod->nz+ibndz-1) 
                            d_tzz[ixs*n1+izs+1] = dd_src2z[isrc];
                    }
                    else if (d_src->orient==5) { /* dipole + - */
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_tzz[(ixs+1)*n1+izs] = dd_src2z[isrc];
                    }
                }
                else if (d_src->type==6) {
                    d_vx[ixs*n1+izs] = dd_src1x[isrc];
                }
                else if (d_src->type==7) {
                    d_vz[ixs*n1+izs] = dd_src1z[isrc];
                }
                
            }
            
        }
        else { /* Elastic scheme */
            
          	if (store) {
                if (verbose>=5) printf("cuda_reSt...ace source at x=%d z=%d restored at free surface\n", ixs, izs);

              	if (d_src->type==1) {
                    if (d_src->orient==1) { /* monopole */
                        d_txx[ixs*n1+izs] = dd_src1x[isrc];
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                    }
                    else if (d_src->orient==2) { /* dipole +/- */
                        d_txx[ixs*n1+izs] = dd_src1x[isrc];
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_txx[ixs*n1+izs+1] = dd_src2x[isrc];
                        d_tzz[ixs*n1+izs+1] = dd_src2z[isrc];
                    }
                    else if (d_src->orient==3) { /* dipole - + */
                        d_txx[ixs*n1+izs] = dd_src1x[isrc];
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_txx[(ixs-1)*n1+izs] = dd_src2x[isrc];
                        d_tzz[(ixs-1)*n1+izs] = dd_src2z[isrc];
                    }
                    else if (d_src->orient==4) { /* dipole +/0/- */
                        if (izs > ibndz) {
                            d_txx[ixs*n1+izs-1] = dd_src1x[isrc];
                            d_tzz[ixs*n1+izs-1] = dd_src1z[isrc];
                        }
                        if (izs < d_mod->nz+ibndz-1) {
                            d_txx[ixs*n1+izs+1] = dd_src1x[isrc];
                            d_tzz[ixs*n1+izs+1] = dd_src1z[isrc];
                        }
                    }
                    else if (d_src->orient==5) { /* dipole + - */
                        d_txx[ixs*n1+izs] = dd_src1x[isrc];
                        d_tzz[ixs*n1+izs] = dd_src1z[isrc];
                        d_txx[(ixs+1)*n1+izs] = dd_src2x[isrc];
                        d_tzz[(ixs+1)*n1+izs] = dd_src2z[isrc];
                    }
              	}
              	else if (d_src->type==2) {
                    
                    /* d_txz source */
                    if ((izs == ibndz) && d_bnd->top==1) {
                        d_txz[(ixs-1)*n1+izs-1] = dd_src1x[isrc];
                        d_txz[ixs*n1+izs-1] = dd_src2x[isrc];
                    }
                    else {
                        d_txz[ixs*n1+izs] = dd_src1x[isrc];
                    }
                    /* possible dipole orientations for a d_txz source */
                    if (d_src->orient == 2) { /* dipole +/- */
                        d_txz[ixs*n1+izs+1] = dd_src2x[isrc];
                    }
                    else if (d_src->orient == 3) { /* dipole - + */
                        d_txz[(ixs-1)*n1+izs] = dd_src2x[isrc];
                    }
                    else if (d_src->orient == 4) { /*  dipole +/O/- */
                        /* correction: subtrace previous value to prevent z-1 values. */
                        d_txz[ixs*n1+izs] = dd_src1x[isrc];
                        d_txz[ixs*n1+izs+1] = dd_src2x[isrc];
                    }
                    else if (d_src->orient == 5) { /* dipole + - */
                        d_txz[(ixs+1)*n1+izs] = dd_src2x[isrc];
                    }
                    
              	}
               	else if (d_src->type==3) {
                   	d_tzz[ixs*n1+izs] = dd_src1z[isrc];
               	}
               	else if (d_src->type==4) {
                   	d_txx[ixs*n1+izs] = dd_src1x[isrc];
               	}
               	else if (d_src->type==5) {
                    
                    d_vx[ixs*n1+izs]= dd_src1x[isrc];
                    d_vz[ixs*n1+izs] = dd_src1z[isrc];
                    d_vx[ixs*n1+izs-1] = dd_src2x[isrc];
                    d_vz[(ixs-1)*n1+izs] = dd_src2z[isrc];
                    
                    /* determine second position of dipole */
                    if (d_src->orient == 2) { /* dipole +/- vertical */
                        izs += 1;
                        d_vx[ixs*n1+izs] = dd_src1x[isrc+d_src->n];
                        d_vz[ixs*n1+izs] = dd_src1z[isrc+d_src->n];
                        d_vx[ixs*n1+izs-1] = dd_src2x[isrc+d_src->n];
                        d_vz[(ixs-1)*n1+izs] = dd_src2z[isrc+d_src->n];
                    }
                    else if (d_src->orient == 3) { /* dipole - + horizontal */
                        ixs += 1;
                        d_vx[ixs*n1+izs] = dd_src1x[isrc+d_src->n];
                        d_vz[ixs*n1+izs] = dd_src1z[isrc+d_src->n];
                        d_vx[ixs*n1+izs-1] = dd_src2x[isrc+d_src->n];
                        d_vz[(ixs-1)*n1+izs] = dd_src2z[isrc+d_src->n];
                    }
                    
				}
               	else if (d_src->type==6) {
                   	d_vx[ixs*n1+izs] = dd_src1x[isrc];
               	}
               	else if (d_src->type==7) {
                   	d_vz[ixs*n1+izs] = dd_src1z[isrc];
               	}
               	else if (d_src->type==8) {
                    
                    d_vx[(ixs+1)*n1+izs] = dd_src1x[isrc];
                    d_vz[ixs*n1+izs+1] = dd_src1z[isrc];
                    d_vx[ixs*n1+izs] = dd_src2x[isrc];
                    d_vz[ixs*n1+izs] = dd_src2z[isrc];
                    
                    /* determine second position of dipole */
                    if (d_src->orient == 2) { /* dipole +/- vertical */
                        izs += 1;
                        d_vx[(ixs+1)*n1+izs] = dd_src1x[isrc+d_src->n];
                        d_vz[ixs*n1+izs+1] = dd_src1z[isrc+d_src->n];
                        d_vx[ixs*n1+izs] = dd_src2x[isrc+d_src->n];
                        d_vz[ixs*n1+izs] = dd_src2z[isrc+d_src->n];
                    }
                    else if (d_src->orient == 3) { /* dipole - + horizontal */
                        ixs += 1;
                        d_vx[(ixs+1)*n1+izs] = dd_src1x[isrc+d_src->n];
                        d_vz[ixs*n1+izs+1] = dd_src1z[isrc+d_src->n];
                        d_vx[ixs*n1+izs] = dd_src2x[isrc+d_src->n];
                        d_vz[ixs*n1+izs] = dd_src2z[isrc+d_src->n];
                    }
                    
               	}
           	}
		}
    }
}
