extern "C"{
	#include<stdlib.h>
	#include<stdio.h>
	#include<math.h>
	#include<assert.h>
	#include"fdelmodc.h"

	//void vmess(char *fmt, ...); // host function cannot be called from GPU kernel
}

#define c1 (9.0/8.0)
#define c2 (-1.0/24.0)

/*********************************************************************
 * 
 * Add's the source amplitude(s) to the grid.
 * 
 * For the acoustic schemes, the source-type must not be txx tzz or txz.
 *
 *   AUTHOR:
 *           Jan Thorbecke (janth@xs4all.nl)
 *           The Netherlands 
 *
 *----------------------------------------------------------------------
 *  GPU version by Victor Koehne, based on 05-2019 version of fdelmodc2D
 *
 *  This is the same function (global kernel), but using global GPU variables
 *	The kernel should be invoked with <<<1,1>>> in this version. 
 *  
 *  Observations: 
 *		- As multiple sources can be injected at the same time, care must be taken 
 *			so that threads do not interfere other thread's injection; for that 
 *			reason, only 1 thread is used. A future (and more optimal) version would 
 *			inject all sources at once and be called as <<<src.n/16, 16>>> (example)
 *
 *		- The 2D array **src_nwav is flattened on the GPU as a 1D array *d_src_nwav

 **********************************************************************/


//int applySource(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float **src_nwav, int verbose)
__global__ void kernel_applySource(modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, int itime, int ixsrc, int izsrc, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_src_nwav, int verbose)
{
	int is0, ibndz, ibndx;
	int isrc, ix, iz, n1;
	int id1, id2;
	float src_ampl, time, scl, dt, sdx;
	float Mxx, Mzz, Mxz, rake;
	static int first=1;
	int i, Nsamp; // d_wav.nsamp[i] accumulator, necessary on multisource case, in order to find correct positions of the flattened array d_src_nwav[ix*Nsamp+it]
	int tid; // thread ID for GPU calculations

	//tid = blockDim.x*blockIdx.x + threadIdx.x; // This version uses only 1 thread, so this is not necessary

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
    	if (d_bnd->top==4 || d_bnd->top==2) ibndz += d_bnd->ntap;
	}
	else {	
    	ibndz = d_mod->ioPz;
    	ibndx = d_mod->ioPx;
    	if (d_bnd->lef==4 || d_bnd->lef==2) ibndx += d_bnd->ntap;
    	if (d_bnd->top==4 || d_bnd->top==2) ibndz += d_bnd->ntap;
	}

	n1   = d_mod->naz;
	dt   = d_mod->dt;
	sdx  = 1.0/d_mod->dx;

	/* special d_txz source activated? */

	if ((d_bnd->top==1) && (d_src->type==2)) {
		iz = izsrc + ibndz;
		if (iz==ibndz) {
            if (d_src->orient != 1) {
				if (first) {
					//vmess("Only monopole Txz source allowed at surface. Reset to monopole");
					printf("cuda_applySource: Only monopole Txz source allowed at surface. Reset to monopole\n");
					first = 0;
				}
				d_src->orient=1;
			}
		}
	}
             
/*
* for plane wave sources the sources are placed 
* around the central shot position 
* the first source position has an offset in x of is0
*
* itime = 0 corresponds with time=0
* itime = 1 corresponds with time=dt
* src[0] (the first sample) corresponds with time = 0
*/

	//is0 = -1*floor((d_src->n-1)/2); //del
	is0 = -1*floorf((d_src->n-1)/2);
#pragma omp	for private (isrc, src_ampl, ix, iz, time, id1, id2, scl) 
	for (isrc=0; isrc<d_src->n; isrc++) {
		src_ampl=0.0;
		/* calculate the source position */
		if (d_src->random || d_src->multiwav) {
			ix = d_src->x[isrc] + ibndx;
			iz = d_src->z[isrc] + ibndz;
		}
		else { /* plane wave and point sources */
            ix = ixsrc + ibndx + is0 + isrc;
            iz = izsrc + ibndz;
		}
		time = itime*dt - d_src->tbeg[isrc];
		id1 = floor(time/dt);
		id2 = id1+1;
        
		/* delay not reached or no samples left in source wavelet? */
		if ( (time < 0.0) || ( (itime*dt) >= d_src->tend[isrc]) ) continue;

//		fprintf(stderr,"isrc=%d ix=%d iz=%d d_src->x=%d d_src->z=%d\n", isrc, ix, iz, d_src->x[isrc], d_src->z[isrc]);

		if (!d_src->multiwav) { /* only one wavelet for all sources */
			//src_ampl = src_nwav[0][id1]*(id2-time/dt) + src_nwav[0][id2]*(time/dt-id1); //del
			src_ampl = d_src_nwav[0*d_wav->nt + id1]*(id2-time/dt) + d_src_nwav[0*d_wav->nt + id2]*(time/dt-id1); 
		}
		else { /* multi-wavelet sources */
			Nsamp = 0;
			for(i=0; i<isrc; i++){
				Nsamp += d_wav->nsamp[i];
			}
			//src_ampl = src_nwav[isrc][id1]*(id2-time/dt) + src_nwav[isrc][id2]*(time/dt-id1); //del
			src_ampl = d_src_nwav[Nsamp + id1]*(id2-time/dt) + d_src_nwav[isrc*Nsamp + id2]*(time/dt-id1);
		}

		if (src_ampl==0.0) continue;
		if ( ((ix-ibndx)<0) || ((ix-ibndx)>d_mod->nx) ) continue; /* source outside grid */

		if (verbose>=4 && itime==0) {
			//vmess("Source %d positioned at grid ix=%d iz=%d",isrc, ix, iz); //del
			printf("cuda_applySource: Source %d positioned at grid ix=%d iz=%d\n",isrc, ix, iz);
		}

		/* cosine squared windowing to reduce edge effects on shot arrays */
		if ( (d_src->n>1) && d_src->window) {
            scl = 1.0;
			if (isrc < d_src->window) {
				scl = cos(0.5*M_PI*(d_src->window - isrc)/d_src->window);
			}
			else if (isrc > d_src->n-d_src->window+1) {
				scl = cos(0.5*M_PI*(d_src->window - (d_src->n-isrc+1))/d_src->window);
			}
			src_ampl *= scl*scl;
		}

		/* source scaling factor to compensate for discretisation */

		/* old amplitude setting does not obey reciprocity */
		// src_ampl *= d_rox[ix*n1+iz]*d_l2m[ix*n1+iz]/(dt);

/* in older version added factor 2.0 to be compliant with defined Green's functions in Marchenko algorithm */
/* this is now set to 1.0 */
		src_ampl *= (1.0/d_mod->dx)*d_l2m[ix*n1+iz];

		if (verbose>5) {
			//vmess("Source %d at grid [ix=%d,iz=%d] at itime %d has value %e",isrc, ix,iz, itime, src_ampl); //del
			printf("cuda_applySource: Source %d at grid [ix=%d,iz=%d] at itime %d has value %e\n",isrc, ix,iz, itime, src_ampl);
		}

		/* Force source */

		if (d_src->type == 6) {
			d_vx[ix*n1+iz] += src_ampl*d_rox[ix*n1+iz]/(d_l2m[ix*n1+iz]);
			/* stable implementation from "Numerical Techniques for Conservation Laws with Source Terms" by Justin Hudson */
			//d_vx[ix*n1+iz] = 0.5*(d_vx[(ix+1)*n1+iz]+d_vx[(ix-1)*n1+iz])+src_ampl*d_rox[ix*n1+iz]/(d_l2m[ix*n1+iz]);
		}
		else if (d_src->type == 7) {
			d_vz[ix*n1+iz] += src_ampl*d_roz[ix*n1+iz]/(d_l2m[ix*n1+iz]);
			/* stable implementation from "Numerical Techniques for Conservation Laws with Source Terms" by Justin Hudson */
			/* stable implementation changes amplitude and more work is needed */
			//d_vz[ix*n1+iz] = 0.5*(d_vz[ix*n1+iz-1]+d_vz[ix*n1+iz+1])+src_ampl*d_roz[ix*n1+iz]/(d_l2m[ix*n1+iz]);
			//d_vz[ix*n1+iz] = 0.25*(d_vz[ix*n1+iz-2]+d_vz[ix*n1+iz-1]+d_vz[ix*n1+iz]+d_vz[ix*n1+iz+1])+src_ampl*d_roz[ix*n1+iz]/(d_l2m[ix*n1+iz]);
        } /* d_src->type */

        
		/* Stress source */

		if (d_mod->ischeme <= 2) { /* Acoustic scheme */
			/* Compressional source */
			if (d_src->type == 1) {
				if (d_src->orient != 1) src_ampl=src_ampl/d_mod->dx;

				if (d_src->orient==1) { /* monopole */
					d_tzz[ix*n1+iz] += src_ampl;
				}
				else if (d_src->orient==2) { /* dipole +/- */
					d_tzz[ix*n1+iz] += src_ampl;
					d_tzz[ix*n1+iz+1] -= src_ampl;
				}
				else if (d_src->orient==3) { /* dipole - + */
					d_tzz[ix*n1+iz] += src_ampl;
					d_tzz[(ix-1)*n1+iz] -= src_ampl;
				}
				else if (d_src->orient==4) { /* dipole +/0/- */
					if (iz > ibndz) 
						d_tzz[ix*n1+iz-1]+= 0.5*src_ampl;
					if (iz < d_mod->nz+ibndz-1) 
						d_tzz[ix*n1+iz+1] -= 0.5*src_ampl;
				}
				else if (d_src->orient==5) { /* dipole + - */
					d_tzz[ix*n1+iz] += src_ampl;
					d_tzz[(ix+1)*n1+iz] -= src_ampl;
				}
			}
		}
		else { /* Elastic scheme */
			/* Compressional source */
			if (d_src->type == 1) {
				if (d_src->orient==1) { /* monopole */
					d_txx[ix*n1+iz] += src_ampl;
					d_tzz[ix*n1+iz] += src_ampl;
				}
				else if (d_src->orient==2) { /* dipole +/- */
					d_txx[ix*n1+iz] += src_ampl;
					d_tzz[ix*n1+iz] += src_ampl;
					d_txx[ix*n1+iz+1] -= src_ampl;
					d_tzz[ix*n1+iz+1] -= src_ampl;
				}
				else if (d_src->orient==3) { /* dipole - + */
					d_txx[ix*n1+iz] += src_ampl;
					d_tzz[ix*n1+iz] += src_ampl;
					d_txx[(ix-1)*n1+iz] -= src_ampl;
					d_tzz[(ix-1)*n1+iz] -= src_ampl;
				}
				else if (d_src->orient==4) { /* dipole +/0/- */
					if (iz > ibndz) {
						d_txx[ix*n1+iz-1]+= 0.5*src_ampl;
						d_tzz[ix*n1+iz-1]+= 0.5*src_ampl;
					}
					if (iz < d_mod->nz+ibndz-1) {
						d_txx[ix*n1+iz+1] -= 0.5*src_ampl;
						d_tzz[ix*n1+iz+1] -= 0.5*src_ampl;
					}
				}
				else if (d_src->orient==5) { /* dipole + - */
					d_txx[ix*n1+iz] += src_ampl;
					d_tzz[ix*n1+iz] += src_ampl;
					d_txx[(ix+1)*n1+iz] -= src_ampl;
					d_tzz[(ix+1)*n1+iz] -= src_ampl;
				}
			}
			else if (d_src->type == 2) {
				/* Txz source */
				if ((iz == ibndz) && d_bnd->top==1) {
					d_txz[(ix-1)*n1+iz-1] += src_ampl;
					d_txz[ix*n1+iz-1] += src_ampl;
				}
				else {
					d_txz[ix*n1+iz] += src_ampl;
				}
				/* possible dipole orientations for a d_txz source */
				if (d_src->orient == 2) { /* dipole +/- */
					d_txz[ix*n1+iz+1] -= src_ampl;
				}
				else if (d_src->orient == 3) { /* dipole - + */
					d_txz[(ix-1)*n1+iz] -= src_ampl;
				}
				else if (d_src->orient == 4) { /*  dipole +/O/- */
					/* correction: subtrace previous value to prevent z-1 values. */
					d_txz[ix*n1+iz] -= 2.0*src_ampl;
					d_txz[ix*n1+iz+1] += src_ampl;
				}
				else if (d_src->orient == 5) { /* dipole + - */
					d_txz[(ix+1)*n1+iz] -= src_ampl;
				}
			}
			/* Tzz source */
			else if(d_src->type == 3) {
				d_tzz[ix*n1+iz] += src_ampl;
			} 
			/* Txx source */
			else if(d_src->type == 4) {
				d_txx[ix*n1+iz] += src_ampl;
			} 

/***********************************************************************
* pure potential shear S source (experimental)
* Curl S-pot = CURL(F) = dF_x/dz - dF_z/dx
***********************************************************************/
			else if(d_src->type == 5) {
				src_ampl = src_ampl*d_rox[ix*n1+iz]/(d_l2m[ix*n1+iz]);
				if (d_src->orient == 3) src_ampl = -src_ampl;
                /* first order derivatives */
				d_vx[ix*n1+iz]     += src_ampl*sdx;
				d_vx[ix*n1+iz-1]   -= src_ampl*sdx;
				d_vz[ix*n1+iz]     -= src_ampl*sdx;
				d_vz[(ix-1)*n1+iz] += src_ampl*sdx;
                
                /* second order derivatives */
                /*
				d_vx[ix*n1+iz]     += c1*src_ampl*sdx;
                d_vx[ix*n1+iz-1]   -= c1*src_ampl*sdx;
				d_vx[ix*n1+iz+1]   += c2*src_ampl*sdx;
                d_vx[ix*n1+iz-2]   -= c2*src_ampl*sdx;

                d_vz[ix*n1+iz]     -= c1*src_ampl*sdx;
				d_vz[(ix-1)*n1+iz] += c1*src_ampl*sdx;
				d_vz[(ix+1)*n1+iz] -= c2*src_ampl*sdx;
				d_vz[(ix-2)*n1+iz] += c2*src_ampl*sdx;
                 */

				/* determine second position of dipole */
				if (d_src->orient == 2) { /* dipole +/- vertical */
					iz += 1;
                    d_vx[ix*n1+iz]     -= src_ampl*sdx;
                    d_vx[ix*n1+iz-1]   += src_ampl*sdx;
                    d_vz[ix*n1+iz]     += src_ampl*sdx;
                    d_vz[(ix-1)*n1+iz] -= src_ampl*sdx;
				}
				else if (d_src->orient == 3) { /* dipole - + horizontal */
					ix += 1;
                    d_vx[ix*n1+iz]     -= src_ampl*sdx;
                    d_vx[ix*n1+iz-1]   += src_ampl*sdx;
                    d_vz[ix*n1+iz]     += src_ampl*sdx;
                    d_vz[(ix-1)*n1+iz] -= src_ampl*sdx;
				}
            }
/***********************************************************************
* pure potential pressure P source (experimental)
* Divergence P-pot = DIV(F) = dF_x/dx + dF_z/dz
***********************************************************************/
            else if(d_src->type == 8) {
			    src_ampl = src_ampl*d_rox[ix*n1+iz]/(d_l2m[ix*n1+iz]);
                if (d_src->orient == 3) src_ampl = -src_ampl;
                d_vx[(ix+1)*n1+iz] += src_ampl*sdx;
                d_vx[ix*n1+iz]     -= src_ampl*sdx;
                d_vz[ix*n1+iz+1]   += src_ampl*sdx;
                d_vz[ix*n1+iz]     -= src_ampl*sdx;
                /* determine second position of dipole */
                if (d_src->orient == 2) { /* dipole +/- */
                    iz += 1;
                    d_vx[(ix+1)*n1+iz] -= src_ampl*sdx;
                    d_vx[ix*n1+iz]     += src_ampl*sdx;
                    d_vz[ix*n1+iz+1]   -= src_ampl*sdx;
                    d_vz[ix*n1+iz]     += src_ampl*sdx;
                }
                else if (d_src->orient == 3) { /* dipole - + */
                    ix += 1;
                    d_vx[(ix+1)*n1+iz] -= src_ampl*sdx;
                    d_vx[ix*n1+iz]     += src_ampl*sdx;
                    d_vz[ix*n1+iz+1]   -= src_ampl*sdx;
                    d_vz[ix*n1+iz]     += src_ampl*sdx;
                }
			}
            else if(d_src->type == 9) {
				rake = 0.5*M_PI;
				// Mxx = -1.0*(sin(d_src->dip)*cos(rake)*sin(2.0*d_src->strike)+sin(d_src->dip*2.0)*sin(rake)*sin(d_src->strike)*sin(d_src->strike)); //del
				// Mxz = -1.0*(cos(d_src->dip)*cos(rake)*cos(d_src->strike)+cos(d_src->dip*2.0)*sin(rake)*sin(d_src->strike)); //del
				// Mzz = sin(d_src->dip*2.0)*sin(rake); //del

				Mxx = -1.0*(sinf(d_src->dip)*cosf(rake)*sinf(2.0*d_src->strike)+sinf(d_src->dip*2.0)*sinf(rake)*sinf(d_src->strike)*sinf(d_src->strike));
				Mxz = -1.0*(cosf(d_src->dip)*cosf(rake)*cosf(d_src->strike)+cosf(d_src->dip*2.0)*sinf(rake)*sinf(d_src->strike));
				Mzz = sinf(d_src->dip*2.0)*sinf(rake);

				d_txx[ix*n1+iz] -= Mxx*src_ampl;
				d_tzz[ix*n1+iz] -= Mzz*src_ampl;
				d_txz[ix*n1+iz] -= Mxz*src_ampl;
			} /* d_src->type */
		} /* ischeme */
	} /* loop over isrc */

	//return 0; //del
}
