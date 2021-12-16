/*
cuda_boundaries
*/
extern "C"{
	#include <stdio.h>
	#include <stdlib.h>
	#include "fdelmodc.h"
}

#include "cuda_myutils.cuh"
#include "cuda_acoustic4.cuh"

/*
// Module variables
float c1, c2;
float dp, dvx, dvz;
// not used in boundariesP, only in boundariesV
// in boundariesV, used for PML
// in boundariesV, used for elastic free surface conditions
// actually, Jan describes boundariesV main purpose as to apply free-surface conditions

int ibnd; //fixed half-stencil size
int ib, ibx, ibz; // think it doesn't need to be declared, make it taper argument

int   nx, nz, n1, n2; // definitely fill this on init

//int   is0, isrc; //not used

int   ixo, ixe, izo, ize; // intensevely used by taper //don't declare and make it argument

// PML variables
int   npml, ipml, pml;
float kappu, alphu, sigmax, R, a, m, fac, dx, dt;
float dpx, dpz, *p;
static float *Vxpml, *Vzpml, *sigmu, *RA;
static int allocated=0;
float Jx, Jz, rho, d;
*/

//////////////////////////
// Module global variables
//////////////////////////
// PML variables
__constant__ static float c1_d, c2_d;
__constant__ static int   n1_d, n2_d, nz_d, nx_d;
__constant__ static float dx_d, dt_d, fac_d;
__constant__ static int   npml_d;
__constant__ static float kappu_d, alphu_d, d_d, sigmax_d, R_d, a_d, m_d;
static float *d_sigmu, *d_RA;
static float  *d_Vxpml, *d_Vzpml, *d_Pxpml, *d_Pzpml;

// CUDA Streams for applying all absorbing boundaries at once
static int nker = 16; // nker = (toplef+toptop+toprig+rigrig+botrig+botbot+botlef+leflef)*2
static int nstreams;
static cudaStream_t *streams; 

// Testing if many streams accelerate taper (it did not)
static int taperkey=1;//del //testing v1's vs. v2's
//////////////////////////////////
// Free surface boundaries kernels
//////////////////////////////////
__global__ void kernel_acoustic_freesurf_vz_top(modPar *d_mod, bndPar *d_bnd, float *d_vz);
__global__ void kernel_acoustic_freesurf_tzz_top_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz);
__global__ void kernel_acoustic_freesurf_tzz_rig_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz);
__global__ void kernel_acoustic_freesurf_tzz_lef_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz);

///////////////////////////
// Rigid boundaries kernels
///////////////////////////
__global__ void kernel_boundaries_rigid_top(modPar *d_mod, float *d_vx, float *d_vz);
__global__ void kernel_boundaries_rigid_bot(modPar *d_mod, float *d_vx, float *d_vz);	
__global__ void kernel_boundaries_rigid_lef(modPar *d_mod, float *d_vx, float *d_vz);
__global__ void kernel_boundaries_rigid_rig(modPar *d_mod, float *d_vx, float *d_vz);
__global__ void kernel_acoustic_freesurf_tzz_bot_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz);

//////////////
// PML kernels
//////////////
/*
Changed Jan's original order so PML is applied as the taper: TOP BOT LEF RIG
Order: toplef toptop toprig
	   botlef botbot botrig
	   leflef rigrig
*/
// Left: leflef toplef botlef
__global__ void kernel_acoustic_pml_leflef_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_leflef_vz_v1(modPar mod, bndPar bnd, float *d_p, float *d_roz, float *d_vz);
__global__ void kernel_acoustic_pml_leflef_p_v1(modPar mod, bndPar bnd, float *d_vx, float *d_vz, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);

__global__ void kernel_acoustic_pml_toplef_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_toplef_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_toplef_p_vx_v1(modPar mod, bndPar bnd, float *d_vx, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);
__global__ void kernel_acoustic_pml_toplef_p_vz_v1(modPar mod, bndPar bnd, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);

__global__ void kernel_acoustic_pml_botlef_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_botlef_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_botlef_p_vz_v1(modPar mod, bndPar bnd, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);
__global__ void kernel_acoustic_pml_botlef_p_vx_v1(modPar mod, bndPar bnd, float *d_vx, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);

// Right: rigrig toprig botrig
__global__ void kernel_acoustic_pml_rigrig_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_rigrig_vz_v1(modPar mod, bndPar bnd, float *d_p, float *d_roz, float *d_vz);
__global__ void kernel_acoustic_pml_rigrig_p_v1(modPar mod, bndPar bnd, float *d_vx, float *d_vz, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);

__global__ void kernel_acoustic_pml_toprig_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_toprig_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_toprig_p_vx_v1(modPar mod, bndPar bnd, float *d_vx, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);
__global__ void kernel_acoustic_pml_toprig_p_vz_v1(modPar mod, bndPar bnd, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);

__global__ void kernel_acoustic_pml_botrig_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_botrig_vx_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vxpml, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_botrig_p_vz_v1(modPar mod, bndPar bnd, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);
__global__ void kernel_acoustic_pml_botrig_p_vx_v1(modPar mod, bndPar bnd, float *d_vx, float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m, float *d_p);


// Top
__global__ void kernel_acoustic_pml_toptop_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_toptop_vx_v1(modPar mod, bndPar bnd, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_toptop_p_v1(modPar mod, bndPar bnd, float *d_vx, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);

// Bot
__global__ void kernel_acoustic_pml_botbot_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu, float *d_Vzpml, float *d_roz, float *d_p, float *d_vz);
__global__ void kernel_acoustic_pml_botbot_vx_v1(modPar mod, bndPar bnd, float *d_rox, float *d_p, float *d_vx);
__global__ void kernel_acoustic_pml_botbot_p_v1(modPar mod, bndPar bnd, float *d_vx, float *d_vz, float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m, float *d_p);


////////////////
// Taper kernels
////////////////
// Top
__global__ void kernel_acoustic_taper_toptop_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_toptop_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

__global__ void kernel_acoustic_taper_toprig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_toprig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

__global__ void kernel_acoustic_taper_toplef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_toplef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

// Bot
__global__ void kernel_acoustic_taper_botbot_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_botbot_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

__global__ void kernel_acoustic_taper_botrig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_botrig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

__global__ void kernel_acoustic_taper_botlef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_botlef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

// Left
__global__ void kernel_acoustic_taper_leflef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_leflef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);

// Right
__global__ void kernel_acoustic_taper_rigrig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, float *d_vx);
__global__ void kernel_acoustic_taper_rigrig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, float *d_vz);


/////////////////////////    
// Init/destroy functions
/////////////////////////
void cuda_init_boundaries(modPar mod, bndPar bnd, int verbose){
/*
Copies constants to CUDA symbol memory (faster access)
*/
	// Staggered grid constants
	float c1, c2;
	int n1, n2, nz, nx;
	float dx, dt, fac;

	c1 = 9.0/8.0;
	c2 = -1.0/24.0;
	nx  = mod.nx;
    nz  = mod.nz;
    n1  = mod.naz;
    n2  = mod.nax;
    dx  = mod.dx;
    dt  = mod.dt;
    fac = dt/dx;

	cudaMemcpyToSymbol(c1_d, &c1, sizeof(float));
	cudaMemcpyToSymbol(c2_d, &c2, sizeof(float));
	wrap_cudaGetLastError("cuda_init_boundaries cudaMemcpyToSymbol");

	// If applicable, init PML
	int   npml, pml_key;
	float kappu, alphu, sigmax, R, a, m;
	float dpx, dpz, *p;
	float *sigmu, *RA;
	float rho, d;

    if ( (bnd.top==2) || (bnd.bot==2) || (bnd.lef==2) || (bnd.rig==2) ) pml_key=1;
	else pml_key=0;

	npml=bnd.npml; /* lenght of pml in grid-points */

    if ( (npml != 0) && pml_key) {
	    sigmu = (float *)calloc(npml,sizeof(float));
	    RA    = (float *)calloc(npml,sizeof(float));
	    
	    /* calculate sigmu and RA only once with fixed velocity Cp */
	    m=bnd.m; /* scaling order */
	    R=bnd.R; /* the theoretical reflection coefficient after discretization */
	    kappu=1.0; /* auxiliary attenuation coefficient for small angles */
	    alphu=0.0;   /* auxiliary attenuation coefficient  for low frequencies */
	    d = (npml-1)*dx; /* depth of pml */
	    /* sigmu attenuation factor representing the loss in the PML depends on the grid position in the PML */
	    
	    sigmax = ((3.0*mod.cp_min)/(2.0*d))*log(1.0/R);
	    for (int ib=0; ib<npml; ib++) { /* ib=0 interface between PML and interior */
	        a = (float) (ib/(npml-1.0));
	        sigmu[ib] = sigmax*pow(a,m);
	        RA[ib] = (1.0)/(1.0+0.5*dt*sigmu[ib]);
	        if (verbose>=3) printf("PML: sigmax=%e cp=%e sigmu[%d]=%e %e\n", sigmax, mod.cp_min, ib, sigmu[ib], a);
	    }

	    cudaMemcpyToSymbol(npml_d, &npml, sizeof(int));
		cudaMemcpyToSymbol(fac_d, &fac, sizeof(float));
		cudaMemcpyToSymbol(dx_d, &dx, sizeof(float));
		cudaMemcpyToSymbol(dt_d, &dt, sizeof(float));
		wrap_cudaGetLastError("cuda_init_boundaries - PML constants");

		cudaMalloc(&d_Vxpml, 2*n1*npml*sizeof(float));
		cudaMalloc(&d_Vzpml, 2*n2*npml*sizeof(float));
		cudaMalloc(&d_Pxpml, 2*n1*npml*sizeof(float));
		cudaMalloc(&d_Pzpml, 2*n2*npml*sizeof(float));
		cudaMalloc(&d_sigmu, npml*sizeof(float));
	    cudaMalloc(&d_RA   , npml*sizeof(float));
		wrap_cudaGetLastError("cuda_init_boundaries - PML cudaMalloc");

		cudaMemset(d_Vxpml, 0, 2*n1*npml*sizeof(float));
		cudaMemset(d_Vzpml, 0, 2*n2*npml*sizeof(float));
		cudaMemset(d_Pxpml, 0, 2*n1*npml*sizeof(float));
		cudaMemset(d_Pzpml, 0, 2*n2*npml*sizeof(float));
		wrap_cudaGetLastError("cuda_init_boundaries - PML cudaMemset");

		cudaMemcpy(d_sigmu, sigmu, npml*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_RA, RA, npml*sizeof(float), cudaMemcpyHostToDevice);
		wrap_cudaGetLastError("cuda_init_boundaries - PML cudaMemcpy");

		free(sigmu);
		free(RA);
	}

	// Testing stuff with MaxMPs and streams
	//wrap_cudaGetDeviceProperties();//del
	cudaDeviceProp prop;
	int idev;
	cudaGetDevice(&idev);
	cudaGetDeviceProperties(&prop, idev);
	int maxthr = prop.multiProcessorCount*prop.maxThreadsPerMultiProcessor;
	int used_thr = 0; // number of threads used by all boundaries kernels (top, bot, lef, rig)
	if (bnd.top==2 || bnd.top==4){ //taper or PML
		used_thr += bnd.ntap*mod.nx; //toptop
		if (bnd.lef==2 || bnd.lef==4) used_thr += bnd.ntap*bnd.ntap; //toplef
		if (bnd.rig==2 || bnd.rig==4) used_thr += bnd.ntap*bnd.ntap; //toprig
	}else{ // rigid, free
		if(bnd.top==1) used_thr += mod.nax;
		if(bnd.top==3) used_thr += mod.nx;
	}
	if (bnd.bot==2 || bnd.bot==4){ //taper or PML
		used_thr += bnd.ntap*mod.nx; //botbot
		if (bnd.lef==2 || bnd.lef==4) used_thr += bnd.ntap*bnd.ntap; //botlef
		if (bnd.rig==2 || bnd.rig==4) used_thr += bnd.ntap*bnd.ntap; //botrig
	}else{ // rigid, free
		if(bnd.bot==1) used_thr += mod.nax;
		if(bnd.bot==3) used_thr += mod.nx;
	}
	if (bnd.lef==2 || bnd.lef==4){ //taper or PML
		used_thr += bnd.ntap*mod.nz; //leflef
	}else{ // rigid, free
		if(bnd.lef==1) used_thr += mod.naz;
		if(bnd.lef==3) used_thr += mod.nz;
	}
	if (bnd.rig==2 || bnd.rig==4){ //taper or PML
		used_thr += bnd.ntap*mod.nz; //rigrig
	}else{ // rigid, free
		if(bnd.lef==1) used_thr += mod.naz;
		if(bnd.lef==3) used_thr += mod.nz;
	}
	used_thr *= 2; // since there are vz and vx kernels
	int fact = int(ceil(float(used_thr)/float(maxthr)));
	nstreams = nker/fact;
	// If used_thr>maxthr no need to use streams, so nstreams=1
	if(nstreams <= 1){
		taperkey = 0; //we test this just for taper
		nstreams = 0;
		if(verbose>3) printf("\t used_thr/maxthr > nkernels (%d/%d=%d > %d)."
			" This means each taper kernel will consume all threads (approx.), so" 
			" there is no gain from running different kernels with different streams." 
			" All kernels will run on the default stream. \n", used_thr, maxthr, fact, 
			nker);
	}else{
		printf("nker=%d fact=%d nstreams=%d\n", nker, fact, nstreams);//del
		if(verbose>3) printfgpu("cuda_init_boundaries\n \tnker = %d\n \t" 
			"maxthr = %d\n \tused_thr = %d\n \tfact = %d\n \tnstreams = %d\n", 
			nker, maxthr, used_thr, fact, nstreams);
		if(verbose>3) printf("Each of the %d kernels uses (roughly) %d threads," 
			" so it is possible to run %d kernels concurrently using %d streams." 
			" These %d streams will run %d times each to complete all %d kernels."
			" The remainder of %d kernel(s) will run on the first stream.\n", nker, 
			used_thr/nker, nstreams, nstreams, nstreams, nker/nstreams, nker, 
			nker%nstreams);

		streams = (cudaStream_t*)malloc(nker*sizeof(cudaStream_t));

		for(int i=0; i<nker; i++){
			if(i<nstreams){
				cudaStreamCreate(&streams[i]);	
			}else{ // streams wrap around nker/fact
				//printf("i=%d nker=%d inew=%d\n", i, nker, i%(nker/fact) );//del
				int ind = i%(nker/fact);
				streams[i] = streams[ind];
				if(verbose>6) printf("Wrapping streams[%d] to streams[%d]\n", i, ind);
			}    	
		}
	}

    // Check for GPU errors
    wrap_cudaGetLastError("cuda_init_boundaries");

    // Output info to user
    if(verbose>1) printfgpu("cuda_init_boundaries.");

}

void cuda_destroy_boundaries(bndPar bnd, int verbose){
/*

*/
	int pml_key;

	if ( (bnd.top==2) || (bnd.bot==2) || (bnd.lef==2) || (bnd.rig==2) ) pml_key=1;
	else pml_key=0;

	// cudaFree PML
	if ( (bnd.npml != 0) && pml_key) {
		cudaFree(d_Vxpml);
		cudaFree(d_Vzpml);
		cudaFree(d_Pxpml);
		cudaFree(d_Pzpml);
		cudaFree(d_sigmu);
	    cudaFree(d_RA);
	}

	// Destroy streams. Remember nkernels make nstreams wraparound
	if(nstreams > 1){
		int istream;
		for(istream=0; istream<nstreams; istream++){
			cudaStreamDestroy(streams[istream]);
		}
		free(streams);
	}

	// Check for GPU errors
    wrap_cudaGetLastError("cuda_destroy_boundaries");

	// Output info to user
    if(verbose>1) printfgpu("cuda_destroy_boundaries.");
}
// end init/destroy functions

///////////////////
// APPLY BOUNDARIES
///////////////////
void cuda_boundariesP(modPar mod, bndPar bnd, modPar *d_mod, bndPar *d_bnd, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul, int itime, int verbose)
{
/*********************************************************************
Does (in order):
1. If applicable, applies top free surface condition to vz field
2. If applicable, applies rigid boundary conditions to vx, vz
3. If applicable, applies PML to acoustic vx, vz
	3.5 If applicable, applies PML to elastic vx, vz



	AUTHOR:
		   Jan Thorbecke (janth@xs4all.nl)
		   The Netherlands 

	GPU version:
			Victor Koehne (ramalhokoehne@gmail.com)
			UFBA/SENAI CIMATEC, Brazil

***********************************************************************/

	int ixo, ixe, izo, ize;
	
	int pml_key=0;
	if ( (bnd.top==2) || (bnd.bot==2) || (bnd.lef==2) || (bnd.rig==2) ) pml_key=1;
	else pml_key=0;

    // Free surface
	if (mod.ischeme <= 2) { /* Acoustic scheme */
		if (bnd.top==1) { /* free surface at top */
		   kernel_acoustic_freesurf_vz_top<<<(mod.nax+255)/256, 256>>>(d_mod, d_bnd, d_vz);
		}
	}

	// Rigid boundaries
	if(bnd.top==3) kernel_boundaries_rigid_top<<<(mod.nx+255)/256, 256, 0, streams[0]>>>(d_mod, d_vx, d_vz);
	if(bnd.bot==3) kernel_boundaries_rigid_bot<<<(mod.nx+255)/256, 256, 0, streams[1]>>>(d_mod, d_vx, d_vz);
	if(bnd.rig==3) kernel_boundaries_rigid_rig<<<(mod.nz+255)/256, 256, 0, streams[2]>>>(d_mod, d_vx, d_vz);
	if(bnd.lef==3) kernel_boundaries_rigid_lef<<<(mod.nz+255)/256, 256, 0, streams[3]>>>(d_mod, d_vx, d_vz);	
 
	/* PML BOUNDARIES (only acoustic case)*/
	if (mod.ischeme == 1 && pml_key) { /* Acoustic scheme PML */
		// Init
		if (itime==0){
			cudaMemset(d_Vxpml, 0, 2*mod.naz*bnd.npml*sizeof(float));
			cudaMemset(d_Vzpml, 0, 2*mod.nax*bnd.npml*sizeof(float));	
		}

		// leflef
		if (bnd.lef == 2) {
			kernel_acoustic_pml_leflef_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
			kernel_acoustic_pml_leflef_vz_v1<<<(mod.naz*mod.nax+255)/256, 256>>>(mod, bnd, d_tzz, d_roz, d_vz);
		}
		// toplef
		if (bnd.lef == 2 && bnd.top == 2){
			kernel_acoustic_pml_toplef_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
			kernel_acoustic_pml_toplef_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);				
		}
		// rigrig
		if (bnd.rig == 2){
			kernel_acoustic_pml_rigrig_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
			kernel_acoustic_pml_rigrig_vz_v1<<<(mod.naz*mod.nax+255)/256, 256>>>(mod, bnd, d_tzz, d_roz, d_vz);			
		}
		// toprig
		if (bnd.top == 2 && bnd.rig == 2) {
			kernel_acoustic_pml_toprig_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
			kernel_acoustic_pml_toprig_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);		
		}
		// toptop
		if (bnd.top == 2){	
			kernel_acoustic_pml_toptop_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);			
			kernel_acoustic_pml_toptop_vx_v1<<<(mod.naz*mod.nax+255)/256, 256>>>(mod, bnd, d_rox, d_tzz, d_vx);
		}
		// botbot
		if (bnd.bot == 2){
			kernel_acoustic_pml_botbot_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);			
			kernel_acoustic_pml_botbot_vx_v1<<<(mod.naz*mod.nax+255)/256, 256>>>(mod, bnd, d_rox, d_tzz, d_vx);
		}	
		// botlef
		if (bnd.bot==2 && bnd.lef == 2){
			kernel_acoustic_pml_botlef_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);
			kernel_acoustic_pml_botlef_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
		}
		// botrig
		if (bnd.bot==2 && bnd.rig == 2){
			kernel_acoustic_pml_botrig_vz_v1<<<(mod.nax+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vzpml, d_roz, d_tzz, d_vz);
			kernel_acoustic_pml_botrig_vx_v1<<<(mod.naz+255)/256,256>>>(mod, bnd, d_RA, d_sigmu, d_Vxpml, d_rox, d_tzz, d_vx);
		}
	} /* end acoustic PML */		


	/* TAPER BOUNDARIES */
	// Top taper: toptop, toprig, toplef	
	if (bnd.top==4) {
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			if(taperkey) kernel_acoustic_taper_toptop_vx_v2<<<(bnd.ntap*mod.nx+255)/256,256,0,streams[0]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(taperkey) kernel_acoustic_taper_toptop_vz_v2<<<(bnd.ntap*mod.nx+255)/256,256,0,streams[1]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

			if(!taperkey) kernel_acoustic_taper_toptop_vx_v2<<<(bnd.ntap*mod.nx+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(!taperkey) kernel_acoustic_taper_toptop_vz_v2<<<(bnd.ntap*mod.nx+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			if(bnd.rig==4){
				if(taperkey) kernel_acoustic_taper_toprig_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[2]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);	
				if(taperkey) kernel_acoustic_taper_toprig_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[3]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

				if(!taperkey) kernel_acoustic_taper_toprig_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);	
				if(!taperkey) kernel_acoustic_taper_toprig_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			} 
			if(bnd.lef==4){
				if(taperkey) kernel_acoustic_taper_toplef_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[4]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);	
				if(taperkey) kernel_acoustic_taper_toplef_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[5]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

				if(!taperkey) kernel_acoustic_taper_toplef_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);	
				if(!taperkey) kernel_acoustic_taper_toplef_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			} 
		}//end acoustic taper
	}// end top taper

	// Bot taper: botbot, botrig, botlef
	if (bnd.bot==4) {
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			if(taperkey) kernel_acoustic_taper_botbot_vx_v2<<<(bnd.ntap*mod.nx+255)/256,256,0,streams[6]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(taperkey) kernel_acoustic_taper_botbot_vz_v2<<<(bnd.ntap*mod.nx+255)/256,256,0,streams[7]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

			if(!taperkey) kernel_acoustic_taper_botbot_vx_v2<<<(bnd.ntap*mod.nx+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(!taperkey) kernel_acoustic_taper_botbot_vz_v2<<<(bnd.ntap*mod.nx+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			if(bnd.rig==4){
				if(taperkey) kernel_acoustic_taper_botrig_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[8]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
				if(taperkey) kernel_acoustic_taper_botrig_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[9]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

				if(!taperkey) kernel_acoustic_taper_botrig_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
				if(!taperkey) kernel_acoustic_taper_botrig_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			}
			if(bnd.lef==4){
				if(taperkey) kernel_acoustic_taper_botlef_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[10]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);					
				if(taperkey) kernel_acoustic_taper_botlef_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256,0,streams[11]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

				if(!taperkey) kernel_acoustic_taper_botlef_vx_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);					
				if(!taperkey) kernel_acoustic_taper_botlef_vz_v2<<<(bnd.ntap*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
			} 
		}
	}//end bot taper

	// Left taper
	if (bnd.lef==4) {
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			if(taperkey) kernel_acoustic_taper_leflef_vx_v2<<<(mod.nz*bnd.ntap+255)/256,256,0,streams[12]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(taperkey) kernel_acoustic_taper_leflef_vz_v2<<<(mod.nz*bnd.ntap+255)/256,256,0,streams[13]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

			if(!taperkey) kernel_acoustic_taper_leflef_vx_v2<<<(mod.nz*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(!taperkey) kernel_acoustic_taper_leflef_vz_v2<<<(mod.nz*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
		}
	}

	// Right taper
	if (bnd.rig==4) {
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			if(taperkey) kernel_acoustic_taper_rigrig_vx_v2<<<(mod.nz*bnd.ntap+255)/256,256,0,streams[14]>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(taperkey) kernel_acoustic_taper_rigrig_vz_v2<<<(mod.nz*bnd.ntap+255)/256,256,0,streams[15]>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);

			if(!taperkey) kernel_acoustic_taper_rigrig_vx_v2<<<(mod.nz*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_rox, d_tzz, d_vx);
			if(!taperkey) kernel_acoustic_taper_rigrig_vz_v2<<<(mod.nz*bnd.ntap+255)/256,256>>>(d_mod, d_bnd, d_roz, d_tzz, d_vz);
		}
	}
}// end cuda_boundariesP

void cuda_boundariesV(modPar mod, bndPar bnd, modPar *d_mod, bndPar *d_bnd, float *d_vx, float *d_vz, float *d_tzz, float *d_txx, float *d_txz, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul, int itime, int verbose)
{
/*********************************************************************
1. If applicable (mod.ischeme==1), applies PML to tzz(P) field
2. If applicable (mod.ischeme==1 && bnd.etc==3), applies free-surface to acoustic tzz(P) field
3. If applicable (mod.ischeme!=1 && bnd.etc==3), applies free-surface to elastic tzz, txz, txx fields
	 
	AUTHOR:
	Jan Thorbecke (janth@xs4all.nl)
	 The Netherlands 

	GPU version:
			Victor Koehne (ramalhokoehne@gmail.com)
			UFBA/SENAI CIMATEC, Brazil

***********************************************************************/

 int pml_key;

 if ( (bnd.top==2) || (bnd.bot==2) || (bnd.lef==2) || (bnd.rig==2) ) pml_key=1;
 else pml_key=0;

	 // PML boundaries (only acoustic case)
	if (mod.ischeme == 1 && pml_key) { /* Acoustic scheme PML */
		if (itime==0){
			cudaMemset(d_Pxpml, 0, 2*mod.naz*bnd.npml*sizeof(float));
			cudaMemset(d_Pzpml, 0, 2*mod.nax*bnd.npml*sizeof(float));
		}
		
		// toptop
		if ( bnd.top == 2) {
			kernel_acoustic_pml_toptop_p_v1<<<(mod.nax+255)/256, 256>>>(mod, bnd, d_vx, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz); 			
		}
		// leflef
 		if (bnd.lef == 2) {
			kernel_acoustic_pml_leflef_p_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_vz, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz); 			
 		}
		// toplef
		if (bnd.top == 2 && bnd.lef == 2) {
			kernel_acoustic_pml_toplef_p_vx_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz);
			kernel_acoustic_pml_toplef_p_vz_v1<<<(mod.nax+255)/256, 256>>>(mod, bnd, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz);				
		}
		// rigrig
 		if (bnd.rig == 2) {
 			kernel_acoustic_pml_rigrig_p_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_vz, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz);
 		}
		// toprig
		if(bnd.top == 2 && bnd.rig == 2){
			kernel_acoustic_pml_toprig_p_vx_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz);
			kernel_acoustic_pml_toprig_p_vz_v1<<<(mod.nax+255)/256, 256>>>(mod, bnd, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz); 				
		}		
 		// botbot
 		if (bnd.bot == 2) {
			kernel_acoustic_pml_botbot_p_v1<<<(mod.naz*mod.nax+255)/256, 256>>>(mod, bnd, d_vx, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz);
		}
		// botrig
		if (bnd.bot == 2 && bnd.rig == 2) {
			kernel_acoustic_pml_botrig_p_vz_v1<<<(mod.nax+255)/256, 256>>>(mod, bnd, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz);
			kernel_acoustic_pml_botrig_p_vx_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz);
		}
		// botlef
		if (bnd.bot == 2 && bnd.lef == 2){
			kernel_acoustic_pml_botlef_p_vz_v1<<<(mod.nax+255)/256, 256>>>(mod, bnd, d_vz, d_RA, d_Pzpml, d_sigmu, d_l2m, d_tzz); 				
			kernel_acoustic_pml_botlef_p_vx_v1<<<(mod.naz+255)/256, 256>>>(mod, bnd, d_vx, d_RA, d_Pxpml, d_sigmu, d_l2m, d_tzz);
		}
	} // end PML boundaries (only acoustic case)

	// FREE SURFACE boundaries
    if (mod.ischeme <= 2) { /* Acoustic scheme */
	    if (bnd.top==1) { /* free surface at top */
		    kernel_acoustic_freesurf_tzz_top_v1<<<(mod.nax+255)/256,256>>>(d_mod, d_bnd, d_tzz);
    	}
    	if (bnd.rig==1){
    		kernel_acoustic_freesurf_tzz_rig_v1<<<(mod.naz+255)/256,256>>>(d_mod, d_bnd, d_tzz);
    	}
    	if (bnd.bot==1){
    		kernel_acoustic_freesurf_tzz_bot_v1<<<(mod.nax+255)/256,256>>>(d_mod, d_bnd, d_tzz);
    	}
    	if (bnd.lef==1){
    		kernel_acoustic_freesurf_tzz_lef_v1<<<(mod.naz+255)/256,256>>>(d_mod, d_bnd, d_tzz);
    	}
    } // end FREE SURFACE boundaries
    else { /* Elastic scheme */
    	printf("cuda_boundaries Elastic scheme free surface not yet implemented! Exiting.\n");
    	exit(0);
	}


}// end cuda_boundariesV


///////////////////////
// FREE SURFACE KERNELS
///////////////////////
__global__ void kernel_acoustic_freesurf_vz_top(modPar *d_mod, bndPar *d_bnd, float *d_vz){
/*
Correctly computes free surface condition at the top of the model
Call: <<<(nax+255)/256,256>>>
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int ix = id; 
	int iz;

	int n1 = d_mod->naz;
	int ixo = d_mod->ioPx;
	int ixe = d_mod->iePx;

    //if(id==0)printf("cuda_boundariesP free surf ixo ixe = %d %d \n",ixo,ixe);//del

	if( ix>=ixo && ix<ixe){
		iz = d_bnd->surface[ix];
		d_vz[ix*n1+iz]   = d_vz[ix*n1+iz+1];
	    d_vz[ix*n1+iz-1] = d_vz[ix*n1+iz+2];		
	}
}


__global__ void kernel_acoustic_freesurf_tzz_top_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz){
/*
Correctly computes free surface condition at the top of the model
Call: <<<(nax+255)/256,256>>>, can be optimized for nx threads
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int ix = id;
	int iz;

	int n1 = d_mod->naz;
	int ixo = d_mod->ioPx;
	int ixe = d_mod->iePx;

	if( ix>=ixo && ix<ixe ){
			iz = d_bnd->surface[ix];
		    d_tzz[ix*n1+iz] = 0.0;
	}
}

__global__ void kernel_acoustic_freesurf_tzz_rig_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz){
/*
Correctly computes free surface condition at the right of the model
Call: <<<(naz+255)/256,256>>>, can be optimized for nz threads
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int ix;
	int iz = id;

	int n1 = d_mod->naz;

	int izo = d_mod->ioPz;
	int ize = d_mod->iePz;

	if( iz>=izo && iz<ize ){
		d_tzz[(d_mod->iePx-1)*n1+iz] = 0.0;
	}
}

__global__ void kernel_acoustic_freesurf_tzz_bot_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz){
/*
Correctly computes free surface condition at the bottom of the model
Call: <<<(nax+255)/256,256>>>, can be optimized for nx threads
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int ix = id;
	int iz;

	int n1 = d_mod->naz;

	int ixo = d_mod->ioPx;
	int ixe = d_mod->iePx;

	// if(id==0) printf("cu freesurf tzz bot ixo ixe = %d %d\n", ixo,ixe);//del

	if( ix>=ixo && ix<ixe ){
		d_tzz[ix*n1 + d_mod->iePz - 1] = 0.0;
	}
}

__global__ void kernel_acoustic_freesurf_tzz_lef_v1(modPar *d_mod, bndPar *d_bnd, float *d_tzz){
/*
Correctly computes free surface condition at the left of the model
Call: <<<(naz+255)/256,256>>>, can be optimized for nz threads
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int ix;
	int iz = id;

	int n1 = d_mod->naz;

	int izo = d_mod->ioPz;
	int ize = d_mod->iePz;

	if( iz>=izo && iz<ize ){
		d_tzz[(d_mod->ioPx-1)*n1+iz] = 0.0;
	}
}



///////////////////////////
// END FREE SURFACE KERNELS
///////////////////////////

/////////////////////////
// RIGID BOUNDARY KERNELS
/////////////////////////
__global__ void kernel_boundaries_rigid_top(modPar *d_mod, float *d_vx, float *d_vz){
/*
Apply rigid boundary at the top
Call: <<<(nx+255)/256, 256>>>
*/	
	int ix = blockIdx.x*blockDim.x + threadIdx.x;

	int n1 = d_mod->naz;
	int nx = d_mod->nx;
	int ibnd = d_mod->iorder/2-1;

	if( ix>=1 && ix<=nx ){
		d_vx[ix*n1+ibnd] = 0.0;
		d_vz[ix*n1+ibnd] = -d_vz[ix*n1+ibnd+1];
		if (d_mod->iorder >= 4) d_vz[ix*n1+ibnd-1] = -d_vz[ix*n1+ibnd+2];
		if (d_mod->iorder >= 6) d_vz[ix*n1+ibnd-2] = -d_vz[ix*n1+ibnd+3];	
	}
	
}

__global__ void kernel_boundaries_rigid_rig(modPar *d_mod, float *d_vx, float *d_vz){
/*
Apply rigid boundary at the right
Call: <<<(nz+255)/256, 256>>>
*/	
	int iz = blockIdx.x*blockDim.x + threadIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;
	int nx = d_mod->nx;
	int ibnd = d_mod->iorder/2-1;

	if( iz>=1 && iz<=nz ){
			d_vz[(nx+ibnd-1)*n1+iz] = 0.0;
			d_vx[(nx+ibnd)*n1+iz]   = -d_vx[(nx+ibnd-1)*n1+iz];
			if (d_mod->iorder == 4) d_vx[(nx+2)*n1+iz] = -d_vx[(nx-1)*n1+iz];
			if (d_mod->iorder == 6) {
				d_vx[(nx+1)*n1+iz] = -d_vx[(nx)*n1+iz];
				d_vx[(nx+3)*n1+iz] = -d_vx[(nx-2)*n1+iz];
			}
	}
	
}


__global__ void kernel_boundaries_rigid_bot(modPar *d_mod, float *d_vx, float *d_vz){
/*
Apply rigid boundary at the bottom
Call: <<<(nx+255)/256, 256>>>
*/	
	int ix = blockIdx.x*blockDim.x + threadIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;
	int nx = d_mod->nx;
	int ibnd = d_mod->iorder/2-1;

	if( ix>=1 && ix<=nx ){
			d_vx[ix*n1+nz+ibnd-1] = 0.0;
			d_vz[ix*n1+nz+ibnd]   = -d_vz[ix*n1+nz+ibnd-1];
			if (d_mod->iorder == 4) d_vz[ix*n1+nz+2] = -d_vz[ix*n1+nz-1];
			if (d_mod->iorder == 6) {
				d_vz[ix*n1+nz+1] = -d_vz[ix*n1+nz];
				d_vz[ix*n1+nz+3] = -d_vz[ix*n1+nz-2];
			}
	}
	
}

__global__ void kernel_boundaries_rigid_lef(modPar *d_mod, float *d_vx, float *d_vz){
/*
Apply rigid boundary at the left
Call: <<<(nz+255)/256, 256>>>
*/	
	int iz = blockIdx.x*blockDim.x + threadIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;
	int nx = d_mod->nx;
	int ibnd = d_mod->iorder/2-1;

	if( iz>=1 && iz<=nz ){
			d_vz[ibnd*n1+iz] = 0.0;
			d_vx[ibnd*n1+iz] = -d_vx[(ibnd+1)*n1+iz];
			if (d_mod->iorder == 4) d_vx[0*n1+iz] = -d_vx[3*n1+iz];
			if (d_mod->iorder == 6) {
				d_vx[1*n1+iz] = -d_vx[4*n1+iz];
				d_vx[0*n1+iz] = -d_vx[5*n1+iz];
			}
	}

}
/////////////////////////////
// END RIGID BOUNDARY KERNELS
/////////////////////////////

//////////////
// PML KERNELS
//////////////

__global__ void kernel_acoustic_pml_leflef_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML left Vx 	
Call: <<<(n1+255)/256, 256>>>

Leftleft PML requires Vxpml accumulation on x-direction, left to right
Therefore, the z-slices are serialized
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ioXx - bnd.npml;
	int ixe  = mod.ioXx;
	int izo  = mod.ioXz;
	int ize  = mod.ieXz;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
					ipml = npml-1-(ix-ixo);
                    rho = (fac_d/d_rox[ix*n1+iz]);
                    dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                          c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
                    Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[iz*npml+ipml]);
                    d_Vxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
                    d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

}


__global__ void kernel_acoustic_pml_leflef_vz_v1(modPar mod, bndPar bnd, float *d_p, float *d_roz, float *d_vz){
/*
PML left vz component; same as acoustic propagation
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized to nz*npml
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	int ix = id/n1;
	int iz = id%n1;
	
	int ixo = mod.ioZx-npml;
	int ixe = mod.ioZx;
	int izo = mod.ioZz;
	int ize = mod.ieZz;

	if(ix>=ixo && ix<ixe && iz>=izo && iz<ize){
	            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
	                            c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
	                            c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
	}

}

__global__ void kernel_acoustic_pml_leflef_p_v1(
								modPar mod, bndPar bnd, float *d_vx, float *d_vz, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML left left P component
Call: <<<(n1+255)/256, 256>>>
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx - npml; 
	int ixe  = mod.ioPx;
	int izo  = mod.ioPz; 
	int ize  = mod.iePz;

	int ix;
	int iz = id;
	int ipml;

	float dvx, dvz, Jx; 

	 // if(id==0) printf("cuboundariesV leflef P-vx ixo ixe izo ize = %d %d %d %d\n", ixo, ixe, izo, ize);

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = npml-1-(ix-ixo);
			
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
		          c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
		    dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
		          c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
		    Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[iz*npml+ipml];
		    d_Pxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
		    d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx+dvz);
		}
	}
}

__global__ void kernel_acoustic_pml_toplef_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML topleft Vx 	
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

TopLeft PML requires Vxpml accumulation on x-direction, left to right
Therefore, processing in the x-dimension is serialized (z in parallel)
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ioXx - npml;
	int ixe  = mod.ioXx;
	int izo  = mod.ioXz - npml;
	int ize  = mod.ioXz;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
					ipml = npml-1-(ix-ixo);
                    rho = (fac_d/d_rox[ix*n1+iz]);
                    dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                          c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
                    Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[iz*npml+ipml]);
                    d_Vxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
                    d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

}

__global__ void kernel_acoustic_pml_toplef_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML TopLeft vz 
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

TopLeft PML of vz requires Vzpml accumulation on z-direction, up->down
Therefore, processing in the z-dimension is serialized (x in parallel)
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	int iz;
	int ix = id;
	int ipml;
	
	int ixo = mod.ioZx-npml;
	int ixe = mod.ioZx;
	int izo = mod.ioZz-npml;
	int ize = mod.ioZz;

	float rho, dpz, Jz;

	if(ix>=ixo && ix<ixe){
        for (iz=izo; iz<ize; iz++) {
        	ipml = npml-1-(iz-izo);
            rho = (fac_d/d_roz[ix*n1+iz]);
            dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
                   c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
            Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[ix*npml+ipml]);
            d_Vzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
    }

}

__global__ void kernel_acoustic_pml_toplef_p_vx_v1(
								modPar mod, bndPar bnd, float *d_vx, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML top left P-vx component
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

TopLeft PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
	// Treat P-limits as vx, vz limits
	int ixo = mod.ioPx - npml; 
	int ixe = mod.ioPx;
	int izo = mod.ioPz - npml;
	int ize = mod.ioPz;

	int ix;
	int iz = id;
	int ipml;

	float dvx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = bnd.npml-1-(ix-ixo);
			
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[iz*npml+ipml];
            d_Pxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx);
		}
	}
}

__global__ void kernel_acoustic_pml_toplef_p_vz_v1(
								modPar mod, bndPar bnd, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML top left P-vz component
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

TopLeft PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx - npml; 
	int ixe  = mod.ioPx;
	int izo  = mod.ioPz - npml;
	int ize  = mod.ioPz;	  

	int ix = id;
	int iz;
	int ipml;

	float dvz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = bnd.npml-1-(iz-izo);

			dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[ix*npml+ipml];
            d_Pzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz);
		}
	}
}

__global__ void kernel_acoustic_pml_botlef_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML BotLeft vz 
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

TopLeft PML of vz requires Vzpml accumulation on z-direction, up->down
Therefore, processing in the z-dimension is serialized (x in parallel)
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;
	int npml = bnd.npml;

	int iz;
	int ix = id;
	int ipml;
	
	int ixo = mod.ioZx-npml;
	int ixe = mod.ioZx;
	int izo = mod.ieZz;
	int ize = mod.ieZz+npml;

	float rho, dpz, Jz;

	if(ix>=ixo && ix<ixe){
        for (iz=izo; iz<ize; iz++) {
        	ipml = (iz-izo);
        	// if(id==ixo) printf("cuboundaries botlef vz ipml=%d\n", ipml);//del
            rho = (fac_d/d_roz[ix*n1+iz]);
            dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
                   c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
            Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[n2*npml+ix*npml+ipml]);
            d_Vzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
    }

	// if(id==ixo) printf("cuboundaries botlef ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}


__global__ void kernel_acoustic_pml_botlef_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML botleft Vx 	
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

BotLeft PML requires Vxpml accumulation on x-direction, left to right
Therefore, processing in the x-dimension is serialized (z in parallel)
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ioXx - npml;
	int ixe  = mod.ioXx;
	int izo  = mod.ieXz;
	int ize  = mod.ieXz + npml;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
					ipml = npml-1-(ix-ixo);
					// if(id==izo) printf("cuboundaries botlef vx ipml=%d\n", ipml);//del
                    rho = (fac_d/d_rox[ix*n1+iz]);
                    dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                          c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
                    Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[iz*npml+ipml]);
                    d_Vxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
                    d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

	// if(id==ixo) printf("cuboundaries botlef vx ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botlef_p_vz_v1(
								modPar mod, bndPar bnd, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML top left P-vz component
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

BotLeft PML of P-vz requires Pzpml accumulation on x-direction, up->down
Therefore z-loop is serialized (x-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx - npml; 
	int ixe  = mod.ioPx;
	int izo  = mod.iePz;
	int ize  = mod.iePz + npml;	  

	int ix = id;
	int iz;
	int ipml;

	float dvz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = (iz-izo);

			dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[n2*npml+ix*npml+ipml];
            d_Pzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz);
		}
	}
}

__global__ void kernel_acoustic_pml_botlef_p_vx_v1(
								modPar mod, bndPar bnd, float *d_vx, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML bot left P-vx component
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

BotLeft PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx - npml; 
	int ixe  = mod.ioPx;
	int izo  = mod.iePz;
	int ize  = mod.iePz + npml;	  

	int ix;
	int iz = id;
	int ipml;

	float dvx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = bnd.npml-1-(ix-ixo);
			
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[iz*npml+ipml];
            d_Pxpml[iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx);
		}
	}
}



__global__ void kernel_acoustic_pml_rigrig_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML right Vx 	
Call: <<<(n1+255)/256, 256>>>; can be optimized to nz threads

RightRight PML requires Vxpml accumulation on x-direction, left to right
Therefore, x-loop is serialized and z-loop parallized
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ieXx;
	int ixe  = mod.ieXx + npml;
	int izo  = mod.ioXz;
	int ize  = mod.ieXz;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = (ix-ixo);
	        rho = (fac_d/d_rox[ix*n1+iz]);
	        dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
	              c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
	        Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[n1*npml+iz*npml+ipml]);
	        d_Vxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
	        d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

}




__global__ void kernel_acoustic_pml_rigrig_vz_v1(modPar mod, bndPar bnd, float *d_p, float *d_roz, float *d_vz){
/*
PML rigrig vz component; same as acoustic propagation
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized to n1*npml
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	int ix = id/n1;
	int iz = id%n1;
	
	int ixo = mod.ieZx;
	int ixe = mod.ieZx + npml;
	int izo = mod.ioZz;
	int ize = mod.ieZz;

	if(ix>=ixo && ix<ixe && iz>=izo && iz<ize){
	            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
	                            c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
	                            c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
	}

}

__global__ void kernel_acoustic_pml_rigrig_p_v1(
								modPar mod, bndPar bnd, float *d_vx, float *d_vz, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML left left P component
Call: <<<(n1+255)/256, 256>>>; could be optimized to nz threads

Requires accumulation of Pxpml variable, lef->rig
Hence, x-loop is serialized and z-loop parallelized
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-lims as vz,vx lims	
	int ixo  = mod.iePx; 
	int ixe  = mod.iePx + npml;
	int izo  = mod.ioPz;
	int ize  = mod.iePz;	  

	int ix;
	int iz = id;
	int ipml;

	float dvx, dvz, Jx; 

	// if(id==0) printf("cuboundariesV toptop P ixo ixe izo ize=%d %d %d %d\n",ixo,ixe,izo,ize);

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = (ix-ixo);
			
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[n1*npml+iz*npml+ipml];
            d_Pxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx+dvz);
		}
	}
}

__global__ void kernel_acoustic_pml_toprig_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML toprig Vx 	
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

TopRig vx PML requires Vxpml accumulation on x-direction, left to right
Therefore, processing in the x-dimension is serialized (z in parallel)
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ieXx;
	int ixe  = mod.ieXx + npml;
	int izo  = mod.ioXz - npml;
	int ize  = mod.ioXz;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = (ix-ixo);
			rho = (fac_d/d_rox[ix*n1+iz]);
            dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                  c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
            Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[n1*npml+iz*npml+ipml]);
            d_Vxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

}

__global__ void kernel_acoustic_pml_toprig_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML TopRig vz 
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

TopRig PML of vz requires Vzpml accumulation on z-direction, up->down
Therefore, processing in the z-dimension is serialized (x in parallel)
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	int iz;
	int ix = id;
	int ipml;
	
	int ixo = mod.ieZx;
	int ixe = mod.ieZx+npml;
	int izo = mod.ioZz-npml;
	int ize = mod.ioZz;

	float rho, dpz, Jz;

	if(ix>=ixo && ix<ixe){
        for (iz=izo; iz<ize; iz++) {
        	ipml = npml-1-(iz-izo);
            rho = (fac_d/d_roz[ix*n1+iz]);
            dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
                   c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
            Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[ix*npml+ipml]);
            d_Vzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
    }

}

__global__ void kernel_acoustic_pml_toprig_p_vx_v1(
								modPar mod, bndPar bnd, float *d_vx, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML top right P-vx component
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

TopRight PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-lims as vz,vx lims	
	int ixo  = mod.iePx; 
	int ixe  = mod.iePx + npml;
	int izo  = mod.ioPz - npml;
	int ize  = mod.ioPz;	  

	int ix;
	int iz = id;
	int ipml;

	float dvx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = (ix-ixo);
			
            dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[n1*npml+iz*npml+ipml];
            d_Pxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx);
		}
	}
}

__global__ void kernel_acoustic_pml_toprig_p_vz_v1(
								modPar mod, bndPar bnd, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML top right P-vz component
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

TopRig PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-lims as vz,vx lims	
	int ixo  = mod.iePx; 
	int ixe  = mod.iePx + npml;
	int izo  = mod.ioPz - npml;
	int ize  = mod.ioPz;	  

	int ix = id;
	int iz;
	int ipml;

	float dvz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = npml-1-(iz-izo);

			dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[ix*npml+ipml];
            d_Pzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz);
		}
	}
}

__global__ void kernel_acoustic_pml_botrig_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML BotRight vz 
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

BotRight PML of vz requires Vzpml accumulation on z-direction, up->down
Therefore, processing in the z-dimension is serialized (x in parallel)
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;
	int npml = bnd.npml;

	int iz;
	int ix = id;
	int ipml;
	
	int ixo = mod.ieZx;
	int ixe = mod.ieZx + npml;
	int izo = mod.ieZz;
	int ize = mod.ieZz + npml;

	float rho, dpz, Jz;

	if(ix>=ixo && ix<ixe){
        for (iz=izo; iz<ize; iz++) {
        	ipml = (iz-izo);
        	// if(id==ixo) printf("cuboundaries botlef vz ipml=%d\n", ipml);//del
		        rho = (fac_d/d_roz[ix*n1+iz]);
		        dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
		               c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
		        Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[n2*npml+ix*npml+ipml]);
		        d_Vzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
		        d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
    }

	// if(id==ixo) printf("cuboundaries botlef ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botrig_vx_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vxpml, float *d_rox, float *d_p, float *d_vx){
/*
PML botrig Vx 	
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

BotRig PML requires Vxpml accumulation on x-direction, left to right
Therefore, processing in the x-dimension is serialized (z in parallel)
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ieXx;
	int ixe  = mod.ieXx + npml;
	int izo  = mod.ieXz;
	int ize  = mod.ieXz + npml;

	int ix;
	int iz = id;
 	int ipml;

	float rho, dpx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
					ipml = (ix-ixo);
					 // if(id==izo) printf("cuboundaries botrig vx ipml=%d\n", ipml);//del
                    rho = (fac_d/d_rox[ix*n1+iz]);
			        dpx = c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
			              c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]);
			        Jx = d_RA[ipml]*(dpx - dt_d*d_Vxpml[n1*npml+iz*npml+ipml]);
			        d_Vxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
			        d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*Jx;
        }
	}

	  // if(id==izo) printf("cuboundaries botrig vx ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botrig_p_vz_v1(
								modPar mod, bndPar bnd, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML bot right P-vz component
Call: <<<(n2+255)/256, 256>>>; can be optimized to npml threads

BotRight PML of P-vz requires Pzpml accumulation on x-direction, up->down
Therefore z-loop is serialized (x-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;

	// Treat P-lims as vz,vx lims	
	int ixo  = mod.iePx; 
	int ixe  = mod.iePx + npml;
	int izo  = mod.iePz;
	int ize  = mod.iePz + npml;

	int ix = id;
	int iz;
	int ipml;

	float dvz, Jz; 


	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = (iz-izo);
			// if(id==ixo) printf("cuboundaries botrig p-vz ipml=%d\n", ipml);//del

			dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
	              c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
	        Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[n2*npml+ix*npml+ipml];
	        d_Pzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
	        d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz);
		}
	}
	// if(id==izo) printf("cuboundaries botrig p-vz ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botrig_p_vx_v1(
								modPar mod, bndPar bnd, float *d_vx, 
								float *d_RA, float *d_Pxpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML bot right P-vx component
Call: <<<(n1+255)/256, 256>>>; can be optimized to npml threads

BotRight PML of P-vx requires Pxpml accumulation on x-direction, lef->rig
Therefore x-loop is serialized (z-loop parallelized)
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
    // Treat P-lims as vz,vx lims
	int ixo  = mod.iePx; 
	int ixe  = mod.iePx + npml;
	int izo  = mod.iePz;
	int ize  = mod.iePz + npml;	  

	int ix;
	int iz = id;
	int ipml;

	float dvx, Jx; 

	if(iz>=izo && iz<ize){
		for (ix=ixo; ix<ixe; ix++) {
			ipml = (ix-ixo);
			// if(id==izo) printf("cuboundaries botrig p-vx ipml=%d\n", ipml);//del			
            dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            Jx = d_RA[ipml]*dvx - d_RA[ipml]*dt_d*d_Pxpml[n1*npml+iz*npml+ipml];
            d_Pxpml[n1*npml+iz*npml+ipml] += d_sigmu[ipml]*Jx;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jx);
		}
	}
	// if(id==izo) printf("cuboundaries botrig p-vx ixo=%d ixe=%d izo=%d ize=%d\n", ixo,ixe,izo,ize);//del
}



__global__ void kernel_acoustic_pml_toptop_vz_v1(modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML TopTop vz 
Call: <<<(n2+255)/256, 256>>>; can be optimized to nx threads

TopTop PML of vz requires Vzpml accumulation on z-direction, up->down
Therefore, processing in the z-dimension is serialized (x in parallel)
*/
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	int iz;
	int ix = id;
	int ipml;
	
	int ixo = mod.ioZx;
	int ixe = mod.ieZx;
	int izo = mod.ioZz-npml;
	int ize = mod.ioZz;

	float rho, dpz, Jz;

	if(ix>=ixo && ix<ixe){
        for (iz=izo; iz<ize; iz++) {
        	ipml = npml-1-(iz-izo);
        	// if(id==ixo) printf("cuda_pml vz ipml=%d\n", ipml);//del
            rho = (fac_d/d_roz[ix*n1+iz]);
            dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
                   c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
            Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[ix*npml+ipml]);
            d_Vzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
    }
    // if(id==0)printf("cuboundariesP vz ixo,ixe,izo,ize=%d,%d,%d,%d\n",ixo,ixe,izo,ize);//del

}

__global__ void kernel_acoustic_pml_toptop_vx_v1(modPar mod, bndPar bnd, 
												float *d_rox, float *d_p, float *d_vx){
/*
PML toptop Vx 	
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized to nx*npml threads

Same as acoustic kernel
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ioXx;
	int ixe  = mod.ieXx;
	int izo  = mod.ioXz - npml;
	int ize  = mod.ioXz;

	int ix = id/n1;
	int iz = id%n1;

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
                                    c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                                    c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]));
	}

 	// if(id==izo)printf("cuda_pml vx ixo,ixe,izo,ize=%d,%d,%d,%d\n",ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_toptop_p_v1(
								modPar mod, bndPar bnd, float *d_vx, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML toptop P component
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized for npml*nx threads

PML toptop P needs variable Pzpml to be accumulated along z, up->down
Hence, z-loop is serialized and x-loop is parallelized
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx; 
	int ixe  = mod.iePx;
	int izo  = mod.ioPz - npml;
	int ize  = mod.ioPz;	  

	int ix = id;
	int iz;
	int ipml;

	float dvx, dvz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = npml-1-(iz-izo);
			 // if(id==ixo) printf("cuda_pml toptop P ipml=%d\n", ipml);//del
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[ix*npml+ipml];
            d_Pzpml[ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz+dvx);
		}
	}
	 // if(id==izo)printf("cuda_pml toptop P ixo,ixe,izo,ize=%d,%d,%d,%d\n",ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botbot_vz_v1( 
								modPar mod, bndPar bnd, float *d_RA, float *d_sigmu,
								float *d_Vzpml, float *d_roz, float *d_p, float *d_vz){
/*
PML botbot Vz
Call: <<<(n2+255)/256, 256>>>; can be optimized to nx threads

Botbot PML requires Vzpml accumulation on z-direction, up-down
Therefore, z-loop is serialized and x-loop parallized
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;

	int npml = bnd.npml;
	int ixo  = mod.ioZx;
	int ixe  = mod.ieZx;
	int izo  = mod.ieZz;
	int ize  = mod.ieZz + npml;

	int ix = id;
	int iz;
 	int ipml;

	float rho, dpz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
					ipml = (iz-izo);
                    rho = (fac_d/d_roz[ix*n1+iz]);
                    dpz = (c1_d*(d_p[ix*n1+iz]   - d_p[ix*n1+iz-1]) +
                           c2_d*(d_p[ix*n1+iz+1] - d_p[ix*n1+iz-2]));
                    Jz = d_RA[ipml]*(dpz - dt_d*d_Vzpml[n2*npml+ix*npml+ipml]);
                    d_Vzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
                    d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*Jz;
        }
	}

}

__global__ void kernel_acoustic_pml_botbot_vx_v1(modPar mod, bndPar bnd, 
												float *d_rox, float *d_p, float *d_vx){
/*
PML botbot Vx 	
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized to nx*npml threads

Same as acoustic kernel
*/		
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;

	int npml = bnd.npml;
	int ixo  = mod.ioXx;
	int ixe  = mod.ieXx;
	int izo  = mod.ieXz;
	int ize  = mod.ieXz + npml;

	int ix = id/n1;
	int iz = id%n1;

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
                            c1_d*(d_p[ix*n1+iz]     - d_p[(ix-1)*n1+iz]) +
                            c2_d*(d_p[(ix+1)*n1+iz] - d_p[(ix-2)*n1+iz]));
	}

 	// if(id==izo)printf("cuda_pml vx ixo,ixe,izo,ize=%d,%d,%d,%d\n",ixo,ixe,izo,ize);//del
}

__global__ void kernel_acoustic_pml_botbot_p_v1(
								modPar mod, bndPar bnd, float *d_vx, float *d_vz, 
								float *d_RA, float *d_Pzpml, float *d_sigmu, float *d_l2m,
								float *d_p){
/*
PML botbot P component (only P-vz)
Call: <<<(n1*n2+255)/256, 256>>>; can be optimized for npml*nx threads

PML botbot P needs variable Pzpml to be accumulated along z, up->down
Hence, z-loop is serialized and x-loop is parallelized
*/	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	int n1 = mod.naz;
	int n2 = mod.nax;
	int npml = bnd.npml;

	// Account for PML (ifs needed because PML may be absent in one of the sides)
    if (bnd.top==2) mod.ioPz += bnd.npml;
    if (bnd.bot==2) mod.iePz -= bnd.npml;
    if (bnd.lef==2) mod.ioPx += bnd.npml;
    if (bnd.rig==2) mod.iePx -= bnd.npml;
	
	// Treat P-limits as vx, vz limits
	int ixo  = mod.ioPx; 
	int ixe  = mod.iePx;
	int izo  = mod.iePz;
	int ize  = mod.iePz + npml;	  

	int ix = id;
	int iz;
	int ipml;

	float dvx, dvz, Jz; 

	if(ix>=ixo && ix<ixe){
		for (iz=izo; iz<ize; iz++) {
			ipml = (iz-izo);
			 // if(id==ixo) printf("cuda_pml toptop P ipml=%d\n", ipml);//del
			dvx = c1_d*(d_vx[(ix+1)*n1+iz] - d_vx[ix*n1+iz]) +
                  c2_d*(d_vx[(ix+2)*n1+iz] - d_vx[(ix-1)*n1+iz]);
            dvz = c1_d*(d_vz[ix*n1+iz+1]   - d_vz[ix*n1+iz]) +
                  c2_d*(d_vz[ix*n1+iz+2]   - d_vz[ix*n1+iz-1]);
            Jz = d_RA[ipml]*dvz - d_RA[ipml]*dt_d*d_Pzpml[n2*npml+ix*npml+ipml];
            d_Pzpml[n2*npml+ix*npml+ipml] += d_sigmu[ipml]*Jz;
            d_p[ix*n1+iz] -= d_l2m[ix*n1+iz]*(Jz+dvx);
		}
	}
	// if(id==izo)printf("cuda_pml botbot P ixo,ixe,izo,ize=%d,%d,%d,%d\n",ixo,ixe,izo,ize);//del
}



/////////////////////
/* END PML KERNELS */
/////////////////////

//////////////////
/* TAPER KERNELS*/
//////////////////

// Start top taper
__global__ void kernel_acoustic_taper_toptop_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, toptop portion
Call: <<<(ntap*nx+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int ntap = d_bnd->ntap;
	int n1 = d_mod->naz;

	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ioXx;
	ixe = d_mod->ieXx;
	izo = d_mod->ioXz - d_bnd->ntap;
	ize = d_mod->ioXz;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	//if(id==0) printf("izo ize ixo ixe = %d %d %d %d \n", izo, ize, ixo, ixe);//del
	// printf("I am thread (iz, ix)=(%d,%d)\n", iz, ix);//del

	ib = (d_bnd->ntap+izo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
			d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
							c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
							c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));

			d_vx[ix*n1+iz]   *= d_bnd->tapx[ib-iz];
			// printf("v2d_vx[%d,%d]=%f\n", iz, ix, d_bnd->tapx[ib-iz]);//del
	}
}//end kernel_acoustic_taper_toptop_vx_v1

__global__ void kernel_acoustic_taper_toptop_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, toptop portion
Call: <<<(ntap*nx+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int ntap = d_bnd->ntap;
	int n1 = d_mod->naz;

	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ioZx;
	ixe = d_mod->ieZx;
	izo = d_mod->ioZz - d_bnd->ntap;
	ize = d_mod->ioZz;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	// if(id==0) printf("AAA ixo ixe izo ize = %d %d %d %d\n",ixo,ixe,izo,ize);//del

	ib = (d_bnd->ntap+izo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
			d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
						c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
						c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));

			d_vz[ix*n1+iz] *= d_bnd->tapz[ib-iz];
	}
}//end kernel_acoustic_taper_toptop_vz_v2

__global__ void kernel_acoustic_taper_toprig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, toprig portion
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibz, ibx;

	ixo = d_mod->ieXx;
	ixe = d_mod->ieXx + ntap;
	izo = d_mod->ioXz - ntap;
	ize = d_mod->ioXz;
	
	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (ntap+izo-1);
	ibx = (ixo);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
								c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
								c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));

		d_vx[ix*n1+iz]   *= d_bnd->tapxz[(ix-ibx)*ntap+(ibz-iz)];	
	}
}// end kernel_acoustic_taper_toprig_vx_v2


__global__ void kernel_acoustic_taper_toprig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, toprig portion
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibz, ibx;

	ixo = d_mod->ieZx;
	ixe = d_mod->ieZx + ntap;
	izo = d_mod->ioZz - ntap;
	ize = d_mod->ioZz;
	
	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (ntap+izo-1);
	ibx = (ixo);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
								c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
								c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));

		d_vz[ix*n1+iz]   *= d_bnd->tapxz[(ix-ibx)*ntap+(ibz-iz)];

	}
}//end kernel_acoustic_taper_toprig_vz_v2


__global__ void kernel_acoustic_taper_toplef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, toplef portion
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibz, ibx;

	ixo = d_mod->ioXx-ntap;
	ixe = d_mod->ioXx;
	izo = d_mod->ioXz-ntap;
	ize = d_mod->ioXz;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (ntap+izo-1);
	ibx = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
								c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
								c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
		
		d_vx[ix*n1+iz]   *= d_bnd->tapxz[(ibx-ix)*ntap+(ibz-iz)];


	}

}//end kernel_acoustic_taper_toplef_vx_v2



__global__ void kernel_acoustic_taper_toplef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, toplef portion
Call: <<<(naz*nax+255)/256,256>>>; can be optimized to ntap*ntap threads
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibz, ibx;

	ixo = d_mod->ioZx - ntap;
	ixe = d_mod->ioZx;
	izo = d_mod->ioZz - ntap;
	ize = d_mod->ioZz;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (ntap+izo-1);
	ibx = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
						d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
									c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
									c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
						
						d_vz[ix*n1+iz]   *= d_bnd->tapxz[(ibx-ix)*ntap+(ibz-iz)];
	}
}//end kernel_acoustic_taper_toplef_vz_v2
// end top taper


// start bot taper
__global__ void kernel_acoustic_taper_botbot_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, botbot portion
Call: <<<(ntap*nx+255)/256,256>>>;
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ioXx;
	ixe = d_mod->ieXx;
	izo = d_mod->ieXz;
	ize = d_mod->ieXz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	// if(id==0) printf("AAA ixo ixe izo ize = %d %d %d %d\n", ixo, ixe, izo, ize);//del
		
	ib = (ize-ntap);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
					c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
		d_vx[ix*n1+iz]   *= d_bnd->tapx[iz-ib];

	}



}// end kernel_acoustic_taper_botbot_vx_v2			


__global__ void kernel_acoustic_taper_botbot_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, botbot portion
Call: <<<(nz*ntap+255)/256,256>>>;
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;
			
	ixo = d_mod->ioZx;
	ixe = d_mod->ieZx;
	izo = d_mod->ieZz;
	ize = d_mod->ieZz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;
	
	ib = (ize-ntap);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
					c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
		d_vz[ix*n1+iz] *= d_bnd->tapz[iz-ib];
	}
}//end kernel_acoustic_taper_botbot_vz_v2


__global__ void kernel_acoustic_taper_botrig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, botrig positions
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibz, ibx;

	ixo = d_mod->ieXx;
	ixe = d_mod->ieXx + ntap;
	izo = d_mod->ieXz;
	ize = d_mod->ieXz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (izo);
	ibx = (ixo);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
						d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
									c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
									c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
	
						d_vx[ix*n1+iz]   *= d_bnd->tapxz[(ix-ibx)*ntap+(iz-ibz)];
	}
}// end kernel_acoustic_taper_botrig_vx_v2


__global__ void kernel_acoustic_taper_botrig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, botrig portion
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibx, ibz;

	ixo = d_mod->ieZx;
	ixe = d_mod->ieZx + ntap;
	izo = d_mod->ieZz;
	ize = d_mod->ieZz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (izo);
	ibx = (ixo);
			

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
					c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
		
		d_vz[ix*n1+iz]   *= d_bnd->tapxz[(ix-ibx)*ntap+(iz-ibz)];
	}

}//end kernel_acoustic_taper_botrig_vz_v2


__global__ void kernel_acoustic_taper_botlef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, botlef positions
Call: <<<(ntap*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibx, ibz;

	ixo = d_mod->ioXx - ntap;
	ixe = d_mod->ioXx;
	izo = d_mod->ieXz;
	ize = d_mod->ieXz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;

	ibz = (izo);
	ibx = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
					c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
		
		d_vx[ix*n1+iz]   *= d_bnd->tapxz[(ibx-ix)*ntap+(iz-ibz)];
	}

}// end kernel_acoustic_taper_botlef_vx_v2


__global__ void kernel_acoustic_taper_botlef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, botrig portion
Call: <<<(naz*nax+255)/256,256>>>; can be optimized to ntap*ntap threads
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ibx, ibz;

	ixo = d_mod->ioZx - ntap;
	ixe = d_mod->ioZx;
	izo = d_mod->ieZz;
	ize = d_mod->ieZz + ntap;

	int iz = id%ntap + izo;
	int ix = id/ntap + ixo;
			
	ibz = (izo);
	ibx = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
		d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
					c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
		
		d_vz[ix*n1+iz]   *= d_bnd->tapxz[(ibx-ix)*ntap+(iz-ibz)];
	}

}// end kernel_acoustic_taper_botlef_vz_v2
// end bot taper



// start left taper
__global__ void kernel_acoustic_taper_leflef_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, botbot portion
Call: <<<(nz*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ioXx-ntap;
	ixe = d_mod->ioXx;
	izo = d_mod->ioXz;
	ize = d_mod->ieXz;

	int iz = id%nz + izo;
	int ix = id/nz + ixo;

	// if(id==0) printf("aaa ixo ixe izo ize = %d %d %d %d\n",ixo,ixe,izo,ize);//del

	ib = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){
			d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
						c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
						c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));
			
			d_vx[ix*n1+iz]   *= d_bnd->tapx[ib-ix];
	}
}//end kernel_acoustic_taper_leflef_vx_v2			


__global__ void kernel_acoustic_taper_leflef_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, leflef portion
Call: <<<(nz*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ioZx - ntap;
	ixe = d_mod->ioZx;
	izo = d_mod->ioZz;
	ize = d_mod->ieZz;

	int iz = id%nz + izo;
	int ix = id/nz + ixo;

	ib = (ntap+ixo-1);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){		
			d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
						c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
						c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));
			
			d_vz[ix*n1+iz] *= d_bnd->tapz[ib-ix];
	}
}//end kernel_leflef_vz_v2


//end left taper

// start right taper
__global__ void kernel_acoustic_taper_rigrig_vx_v2(modPar *d_mod, bndPar *d_bnd, float *d_rox, float *d_tzz, 
									float *d_vx){
/*
Applies taper to vx, rigrig portion
Call: <<<(nz*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ieXx;
	ixe = d_mod->ieXx + ntap;
	izo = d_mod->ioXz;
	ize = d_mod->ieXz;

	int iz = id%nz + izo;
	int ix = id/nz + ixo;
		
	ib = (ixe-ntap);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){		
		d_vx[ix*n1+iz] -= d_rox[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]	 - d_tzz[(ix-1)*n1+iz]) +
					c2_d*(d_tzz[(ix+1)*n1+iz] - d_tzz[(ix-2)*n1+iz]));

		d_vx[ix*n1+iz]   *= d_bnd->tapx[ix-ib];
	}
	// if(iz == izo && ix == ixo){
	// 	for(int idx = ixo; idx<ixe; idx++)
	// 		printf("cuda_boundaries taper vx vx bnd.tapx[%d] = %f\n",idx-ib,d_bnd->tapx[idx-ib]);//del	
	// } 
}// end kernel_acoustic_taper_rigrig_vx_v2


__global__ void kernel_acoustic_taper_rigrig_vz_v2(modPar *d_mod, bndPar *d_bnd, float *d_roz, float *d_tzz, 
									float *d_vz){
/*
Applies taper to vz, rigrig portion
Call: <<<(nz*ntap+255)/256,256>>>
*/	
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	int n1 = d_mod->naz;
	int nz = d_mod->nz;

	int ntap = d_bnd->ntap;
	int ixo, ixe, izo, ize, ib;

	ixo = d_mod->ieZx;
	ixe = d_mod->ieZx + ntap;
	izo = d_mod->ioZz;
	ize = d_mod->ieZz;

	int iz = id%nz + izo;
	int ix = id/nz + ixo;
			
	ib = (ixe-ntap);

	if(iz>=izo && iz<ize && ix>=ixo && ix<ixe){		
		d_vz[ix*n1+iz] -= d_roz[ix*n1+iz]*(
					c1_d*(d_tzz[ix*n1+iz]   - d_tzz[ix*n1+iz-1]) +
					c2_d*(d_tzz[ix*n1+iz+1] - d_tzz[ix*n1+iz-2]));

		d_vz[ix*n1+iz] *= d_bnd->tapz[ix-ib];
	}
	// if(iz == izo && ix == ixo){
	// 	printf("passed for iz=%d ix=%d\n", iz, ix);
	// 	for(int idx = ixo; idx<ixe; idx++)
	// 		printf("cuda_boundaries taper vz bnd.tapz[%d] = %f\n",idx-ib,d_bnd->tapz[idx-ib]);//del	
	// } 

}// end kernel_acoustic_taper_rigrig_vz_v2


// end right taper


//////////////////////
/* END TAPER KERNELS*/
//////////////////////
