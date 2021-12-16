#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include"fdelmodc.h"

#define MIN(x,y) ((x) < (y) ? (x) : (y))

int applySource(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float **src_nwav, int verbose);

int storeSourceOnSurface(modPar mod, srcPar src, bndPar bnd, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, int verbose);

int reStoreSourceOnSurface(modPar mod, srcPar src, bndPar bnd, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, int verbose);

int boundariesP(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);

int boundariesV(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);

int boundariesP2(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);//del
// //del
int boundariesV2(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);//del
// //del

int acoustic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose)
{
/*********************************************************************
       COMPUTATIONAL OVERVIEW OF THE 4th ORDER STAGGERED GRID: 

  The captial symbols T (=P) Txz,Vx,Vz represent the actual grid
  The indices ix,iz are related to the T grid, so the capital T 
  symbols represent the actual modelled grid.

  one cel (iz,ix)
       |
       V                              extra column of vx,txz
                                                      |
    -------                                           V
   | txz vz| txz vz  txz vz  txz vz  txz vz  txz vz txz
   |       |      
   | vx  t | vx  t   vx  t   vx  t   vx  t   vx  t  vx
    -------
     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz

     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx
                 |   |   |   |   |   |   | 
     txz vz  txz Vz--Txz-Vz--Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx
                 |   |   |   |   |   |   |
     txz vz  txz Vz  Txz-Vz  Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx
                 |   |   |   |   |   |   |
     txz vz  txz Vz  Txz-Vz  Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx

     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz

     vx  t   vx  t   vx  t   vx  t   vx  t   vx  t  vx

     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz  <--| 
                                                             |
                                         extra row of txz/vz |

   AUTHOR:
           Jan Thorbecke (janth@xs4all.nl)
           The Netherlands 

***********************************************************************/

	float c1, c2;
	int   ix, iz;
	int   n1;
	int   ioXx, ioXz, ioZz, ioZx, ioPx, ioPz;

	c1 = 9.0/8.0; 
	c2 = -1.0/24.0;
	n1  = mod.naz;

/*
	ioXx=mod.iorder/2;
	ioXz=ioXx-1;
	ioZz=mod.iorder/2;
	ioZx=ioZz-1;
	ioPx=mod.iorder/2-1;
	ioPz=ioPx;
*/
	/* calculate vx for all grid points except on the virtual boundary*/
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
#pragma ivdep
		for (iz=mod.ioXz; iz<mod.ieXz; iz++) {
			vx[ix*n1+iz] -= rox[ix*n1+iz]*(
				c1*(p[ix*n1+iz]     - p[(ix-1)*n1+iz]) +
				c2*(p[(ix+1)*n1+iz] - p[(ix-2)*n1+iz]));
		}
	}

	/* calculate vz for all grid points except on the virtual boundary */
#pragma omp for private (ix, iz) schedule(guided,1) 
	for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
#pragma ivdep
		for (iz=mod.ioZz; iz<mod.ieZz; iz++) {
			vz[ix*n1+iz] -= roz[ix*n1+iz]*(
						c1*(p[ix*n1+iz]   - p[ix*n1+iz-1]) +
						c2*(p[ix*n1+iz+1] - p[ix*n1+iz-2]));
		}
	}
        
	/* boundary condition clears velocities on boundaries */
	 // printf("acoustic4 pre-boundariesP mod.ioXx = %d \tmod.ieXx = %d\n", mod.ioXx, mod.ieXx);// del
	 // printf("acoustic4 pre-boundariesP mod.ioZx = %d \tmod.ieZx = %d\n", mod.ioZx, mod.ieZx);// del
	 // printf("exit after 1 run\n");//del
	 // exit(0);
	boundariesP(mod, bnd, vx, vz, p, NULL, NULL, rox, roz, l2m, NULL, NULL, itime, verbose);
	//boundariesP2(mod, bnd, vx, vz, p, NULL, NULL, rox, roz, l2m, NULL, NULL, itime, verbose);//del

	/* Add force source */
	if (src.type > 5) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, p, NULL, NULL, rox, roz, l2m, src_nwav, verbose);
	}

	/* this is needed because the P fields are not using tapered boundaries (bnd....=4) */
	// printf("acoustic4 (before) mod.ioPx = %d \tmod.iePx = %d\n", mod.ioPx, mod.iePx);// del
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
    	 // printf("acoustic4 P ixo=%d ixe=%d izo=%d ize=%d\n", mod.ioPx, mod.iePx, mod.ioPz, mod.iePz);//del
	   // printf("acoustic4 pre-prop mod.ioPx = %d \tmod.iePx = %d\n", mod.ioPx, mod.iePx);// del
	   // printf("acoustic4 pre-prop mod.ioPz = %d \tmod.iePz = %d\n", mod.ioPz, mod.iePz);// del
	   // exit(0);
	/* calculate p/tzz for all grid points except on the virtual boundary */
#pragma omp for private (ix, iz) schedule(guided,1) 
//#pragma omp for private (ix, iz) schedule(dynamic) 
#pragma ivdep
	for (ix=mod.ioPx; ix<mod.iePx; ix++) {
#pragma ivdep
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			p[ix*n1+iz] -= l2m[ix*n1+iz]*(
						c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
						c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]) +
						c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
						c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]));
		}
	}
    if (bnd.top==2) mod.ioPz -= bnd.npml;
    if (bnd.bot==2) mod.iePz += bnd.npml;
    if (bnd.lef==2) mod.ioPx -= bnd.npml;
    if (bnd.rig==2) mod.iePx += bnd.npml;
	 // printf("acoustic4 (after) mod.ioPx = %d \tmod.iePx = %d\n", mod.ioPx, mod.iePx);// del
	 // exit(0);//we are here del

	/* Add stress source */
	if (src.type < 6) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, p, NULL, NULL, rox, roz, l2m, src_nwav, verbose);
	}
    
/* Free surface: calculate free surface conditions for stresses */

	/* check if there are sources placed on the free surface */
    storeSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, p, NULL, NULL, verbose);

	/* Free surface: calculate free surface conditions for stresses */
	 // printf("acoustic4 pre-boundariesV mod.ioXx = %d \tmod.ieXx = %d\n", mod.ioXx, mod.ieXx);// del
	 // printf("acoustic4 pre-boundariesV mod.ioZx = %d \tmod.ieZx = %d\n", mod.ioZx, mod.ieZx);// del
	boundariesV(mod, bnd, vx, vz, p, NULL, NULL, rox, roz, l2m, NULL, NULL, itime, verbose);
	//boundariesV2(mod, bnd, vx, vz, p, NULL, NULL, rox, roz, l2m, NULL, NULL, itime, verbose);//del
	// exit(0); //del

	/* restore source positions on the edge */
	reStoreSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, p, NULL, NULL, verbose);

	return 0;
}
