/*
Module to alloc and copyH2D fdelmodc2D parameters on GPU
Contains functions to deep free the structures as well
Contains auxiliary functions to print structures on host and device as well.
*/

#include "fdelmodc.h" //for the structure definitions
#include "string.h"
#include "cuda_myutils.cuh"


void cuda_deepcpy_modPar(modPar h_mod, modPar *d_mod);
void cudaFree_modPar(modPar *d_mod);
void cuda_deepcpy_wavPar(wavPar h_wav, srcPar h_src, wavPar *d_wav);
void cudaFree_wavPar(wavPar *d_wav);
void cuda_deepcpy_srcPar(srcPar h_src, srcPar *d_src);
void cudaFree_srcPar(srcPar *d_src);
void cuda_deepcpy_recPar(recPar h_rec, recPar *d_rec);
void cudaFree_recPar(recPar *d_rec);
void cuda_deepcpy_snaPar(snaPar h_sna, snaPar *d_sna);
void cudaFree_snaPar(snaPar *d_sna);
void cuda_deepcpy_shotPar(shotPar h_shot, shotPar *d_shot);
void cudaFree_shotPar(shotPar *d_shot);
void cuda_deepcpy_bndPar(bndPar h_bnd, modPar h_mod, bndPar *d_bnd);
void cudaFree_bndPar(bndPar *d_bnd);

void printf_modPar(modPar mod);
__global__ void kernel_printf_modPar(modPar *d_mod);
void printf_wavPar(wavPar wav);
__global__ void kernel_printf_wavPar(wavPar *d_wav);
void printf_srcPar(srcPar src);
__global__ void kernel_printf_srcPar(srcPar *d_src);
void printf_recPar(recPar rec);
__global__ void kernel_printf_recPar(recPar *d_rec);
void printf_snaPar(snaPar sna);
__global__ void kernel_printf_snaPar(snaPar *d_sna);
void printf_shotPar(shotPar shot);
__global__ void kernel_printf_shotPar(shotPar *d_shot);
void printf_bndPar(bndPar bnd);
__global__ void kernel_printf_bndPar(bndPar *d_bnd);


void cuda_initParStructs(modPar mod, srcPar src, wavPar wav, bndPar bnd, 
					recPar rec, shotPar shot, snaPar sna,
					modPar **d_mod, srcPar **d_src, wavPar **d_wav, bndPar **d_bnd, 
					recPar **d_rec, shotPar **d_shot, snaPar **d_sna, int verbose){

	cudaMalloc(d_mod, sizeof(modPar)); 
	cudaMalloc(d_rec, sizeof(recPar));
	cudaMalloc(d_sna, sizeof(snaPar));
	cudaMalloc(d_src, sizeof(srcPar));
	cudaMalloc(d_wav, sizeof(wavPar));
	cudaMalloc(d_shot, sizeof(shotPar));
	cudaMalloc(d_bnd, sizeof(bndPar));
	wrap_cudaGetLastError("cuda_initParStructs:error cudaMallocs");

	cuda_deepcpy_modPar(mod, *d_mod);
	cuda_deepcpy_srcPar(src, *d_src);
	cuda_deepcpy_wavPar(wav, src, *d_wav);
	cuda_deepcpy_bndPar(bnd, mod, *d_bnd); 
	cuda_deepcpy_recPar(rec, *d_rec);
	cuda_deepcpy_shotPar(shot, *d_shot);
	cuda_deepcpy_snaPar(sna, *d_sna);
	wrap_cudaGetLastError("cuda_initParStructs: deepcpys");

	if(verbose>=1) printfgpu("cuda_initParStructs.");
}

void cuda_freeParStructs(modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, 
						 recPar *d_rec, shotPar *d_shot, snaPar *d_sna, int verbose){

	cudaFree_modPar(d_mod);
	cudaFree_srcPar(d_src);
	cudaFree_wavPar(d_wav);
	cudaFree_bndPar(d_bnd);
	cudaFree_recPar(d_rec);
	cudaFree_shotPar(d_shot);
	cudaFree_snaPar(d_sna);

	wrap_cudaGetLastError("cuda_freeParStructs: failed.");	
	if(verbose>=1) printfgpu("cuda_freeParStructs.");
}

void cuda_initCoefFields(modPar mod, float *rox, float *roz, float *l2m, float *lam, float *mul,
						float *tss, float *tes, float *tep, float *p, float *q, float *r,
						float **d_rox, float **d_roz, float **d_l2m, float **d_lam, float **d_mul,
						float **d_tss, float **d_tes, float **d_tep, float **d_p, float **d_q, 
						float **d_r, int verbose){
/*
Initialize coefficients fields on the GPU.
Arguments are ptr to ptr (**) because cudaMalloc needs the address of their address.
*/
	int sizem=mod.nax*mod.naz;
	// Allocation
	cudaMalloc(d_rox, sizem*sizeof(float));
	cudaMalloc(d_roz, sizem*sizeof(float));
	cudaMalloc(d_l2m, sizem*sizeof(float));
	if (mod.ischeme==2) {
		cudaMalloc(d_tss, sizem*sizeof(float));
		cudaMalloc(d_tep, sizem*sizeof(float));
		cudaMalloc(d_q, sizem*sizeof(float));
	}
	if (mod.ischeme>2) {
		cudaMalloc(d_lam, sizem*sizeof(float));
		cudaMalloc(d_mul, sizem*sizeof(float));
	}
	if (mod.ischeme==4) {
		cudaMalloc(d_tss, sizem*sizeof(float));
		cudaMalloc(d_tes, sizem*sizeof(float));
		cudaMalloc(d_tep, sizem*sizeof(float));
		cudaMalloc(d_r, sizem*sizeof(float));
		cudaMalloc(d_p, sizem*sizeof(float));
		cudaMalloc(d_q, sizem*sizeof(float));
	}
	wrap_cudaGetLastError("cuda_initCoefFields: cudaMalloc");

	// Initialize by copying from host vars
	cudaMemcpy(*d_rox, rox, sizem*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_roz, roz, sizem*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_l2m, l2m, sizem*sizeof(float), cudaMemcpyHostToDevice);
	if (mod.ischeme==2) {
		cudaMemcpy(*d_tss, tss, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_tep, tep, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_q, q, sizem*sizeof(float), cudaMemcpyHostToDevice);
	}
	if (mod.ischeme>2) {
		cudaMemcpy(*d_lam, lam, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_mul, mul, sizem*sizeof(float), cudaMemcpyHostToDevice);
	}
	if (mod.ischeme==4) {
		cudaMemcpy(*d_tss, tss, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_tes, tes, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_tep, tep, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_r, r, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_p, p, sizem*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(*d_q, q, sizem*sizeof(float), cudaMemcpyHostToDevice);
	}
	wrap_cudaGetLastError("cuda_initCoefFields: cudaMemcpy's");

	if(verbose>=1) printfgpu("cuda_initCoefFields.");
}

void cuda_freeCoefFields(modPar mod, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul,
						float *d_tss, float *d_tes, float *d_tep, float *d_p, float *d_q, float *d_r, int verbose){
/*
Cuda free's coeffiecients fields.
Arguments are single pointers (*) since cudaFree does not need the address of their address.
*/

	cudaFree(d_rox);
	cudaFree(d_roz);
	cudaFree(d_l2m);
	if (mod.ischeme==2) {
		cudaFree(d_tss);
		cudaFree(d_tep);
		cudaFree(d_q);
	}
	if (mod.ischeme>2) {
		cudaFree(d_lam);
		cudaFree(d_mul);
	}
	if (mod.ischeme==4) {
		cudaFree(d_tss);
		cudaFree(d_tes);
		cudaFree(d_tep);
		cudaFree(d_r);
		cudaFree(d_p);
		cudaFree(d_q);
	}
	wrap_cudaGetLastError("cuda_freeCoefFields");

	if(verbose>=1) printfgpu("cuda_freeCoefFields.");
}

void cuda_initMainFields(modPar mod, float **d_vx, float **d_vz, float **d_tzz, float **d_txz, float **d_txx, int verbose){
/*
Init main fields vx, vz, etc on the GPU.
The fields are ptr2ptr's (**) because cudaMalloc needs their addresses
*/
	int sizem = mod.nax*mod.naz;

	cudaMalloc(d_vx, sizem*sizeof(float));
	cudaMalloc(d_vz, sizem*sizeof(float));
	cudaMalloc(d_tzz, sizem*sizeof(float)); /* =P field for acoustic */
	if (mod.ischeme>2) {
		cudaMalloc(d_txz, sizem*sizeof(float));
		cudaMalloc(d_txx, sizem*sizeof(float));
	}

	wrap_cudaGetLastError("cuda_initMainFields cudaMalloc");

	cudaMemset(*d_vx, 0, sizem*sizeof(float));
	cudaMemset(*d_vz, 0, sizem*sizeof(float));
	cudaMemset(*d_tzz, 0, sizem*sizeof(float)); /* =P field for acoustic */
	if (mod.ischeme>2) {
		cudaMemset(*d_txz, 0, sizem*sizeof(float));
		cudaMemset(*d_txx, 0, sizem*sizeof(float));
	}

	wrap_cudaGetLastError("cuda_initMainFields cudaMemset");

	if(verbose>=1) printfgpu("cuda_initMainFields.");
}

void cuda_freeMainFields(modPar mod, float *d_vx, float *d_vz, float *d_tzz, float *d_txz, float *d_txx, int verbose){
/*
Frees main fields on the GPU
*/
	cudaFree(d_vx);
	cudaFree(d_vz);
	cudaFree(d_tzz);
	
	if (mod.ischeme>2) {
		cudaFree(d_txz);
		cudaFree(d_txx);
	}

	wrap_cudaGetLastError("cuda_freeMainFields");

	if(verbose>=1) printfgpu("cuda_freeMainFields.");
}

void cuda_initRec(modPar mod, recPar rec, float **d_rec_vx, float **d_rec_vz, float **d_rec_p, float **d_rec_txx, 
				  float **d_rec_tzz, float **d_rec_txz, float **d_rec_pp, float **d_rec_ss, 
				  float **d_rec_udp, float **d_rec_udvz, int verbose){
/*
Init receiver arrays on the GPU.
Arguments are ptr2ptr (**) because cudaMalloc needs the address of their address.
*/

	int size = rec.n*rec.nt;

	if (rec.type.vz)  cudaMalloc(d_rec_vz, size*sizeof(float));
	if (rec.type.vx)  cudaMalloc(d_rec_vx, size*sizeof(float));
	if (rec.type.p)   cudaMalloc(d_rec_p, size*sizeof(float));
	if (rec.type.txx) cudaMalloc(d_rec_txx, size*sizeof(float));
	if (rec.type.tzz) cudaMalloc(d_rec_tzz, size*sizeof(float));
	if (rec.type.txz) cudaMalloc(d_rec_txz, size*sizeof(float));
	if (rec.type.pp)  cudaMalloc(d_rec_pp, size*sizeof(float));
	if (rec.type.ss)  cudaMalloc(d_rec_ss, size*sizeof(float));
    if (rec.type.ud) { 
		cudaMalloc(d_rec_udvz, mod.nax*rec.nt*sizeof(float));
		cudaMalloc(d_rec_udp, mod.nax*rec.nt*sizeof(float));
	}

	wrap_cudaGetLastError("cuda_initRec: cudaMalloc");

	if (rec.type.vz)  cudaMemset(*d_rec_vz, 0, size*sizeof(float));
	if (rec.type.vx)  cudaMemset(*d_rec_vx, 0, size*sizeof(float));
	if (rec.type.p)   cudaMemset(*d_rec_p, 0, size*sizeof(float));
	if (rec.type.txx) cudaMemset(*d_rec_txx, 0, size*sizeof(float));
	if (rec.type.tzz) cudaMemset(*d_rec_tzz, 0, size*sizeof(float));
	if (rec.type.txz) cudaMemset(*d_rec_txz, 0, size*sizeof(float));
	if (rec.type.pp)  cudaMemset(*d_rec_pp, 0, size*sizeof(float));
	if (rec.type.ss)  cudaMemset(*d_rec_ss, 0, size*sizeof(float));
    if (rec.type.ud) { 
		cudaMemset(*d_rec_udvz, 0, mod.nax*rec.nt*sizeof(float));
		cudaMemset(*d_rec_udp, 0, mod.nax*rec.nt*sizeof(float));
	}

	wrap_cudaGetLastError("cuda_initRec: cudaMemset");

	if(verbose>=1) printfgpu("cuda_initRec.");
}

void cuda_freeRec(recPar rec, float *d_rec_vx, float *d_rec_vz, float *d_rec_p, float *d_rec_txx, 
				  float *d_rec_tzz, float *d_rec_txz, float *d_rec_pp, float *d_rec_ss, 
				  float *d_rec_udp, float *d_rec_udvz, int verbose){
/*
Frees rec variables on the GPU.
*/

	if (rec.type.vz)  cudaFree(d_rec_vz);
	if (rec.type.vx)  cudaFree(d_rec_vx);
	if (rec.type.p)   cudaFree(d_rec_p);
	if (rec.type.txx) cudaFree(d_rec_txx);
	if (rec.type.tzz) cudaFree(d_rec_tzz);
	if (rec.type.txz) cudaFree(d_rec_txz);
	if (rec.type.pp)  cudaFree(d_rec_pp);
	if (rec.type.ss)  cudaFree(d_rec_ss);
    if (rec.type.ud) { 
		cudaFree(d_rec_udvz);
		cudaFree(d_rec_udp);
	}

	wrap_cudaGetLastError("cuda_freeRec");
	
	if(verbose>=1) printfgpu("cuda_freeRec.");
}

void cuda_init_d_src_nwav(wavPar wav, float **src_nwav, float **d_src_nwav, int verbose){
/*
Init d_src_nwav on the GPU.
d_src_nwav is a ptr2ptr just for cudaMalloc.
It is allocated as an 1d-array on the GPU.
*/
	int ix, isamp;
	int n1;


	if (wav.random) {
		cudaMalloc(d_src_nwav, wav.nst*sizeof(float));

		isamp = 0;
		for(ix=0; ix<wav.nx; ix++){
			n1 = wav.nsamp[ix];
			cudaMemcpy(&d_src_nwav[0][isamp], &src_nwav[ix][0], n1*sizeof(float), cudaMemcpyHostToDevice);
			isamp += n1;
		}
	}
	else {
		n1 = wav.nsamp[0];

		cudaMalloc(d_src_nwav, wav.nt*wav.nx*sizeof(float));
		cudaMemcpy(&d_src_nwav[0][0], &src_nwav[0][0], n1*sizeof(float), cudaMemcpyHostToDevice);
	}
	wrap_cudaGetLastError("cuda_init_d_src_nwav malloc");

	if(verbose>=1) printfgpu("cuda_init_d_src_nwav.");
}

void cuda_free_d_src_nwav(float *d_src_nwav, int verbose){
	cudaFree(d_src_nwav);
	wrap_cudaGetLastError("cuda_free_d_src_nwav");	

	if(verbose>=1) printfgpu("cuda_free_d_src_nwav.");
}


void cuda_deepcpy_modPar(modPar h_mod, modPar *d_mod){
/*
"deep copy" h_mod to d_mod
*/
	// single variables
	cudaMemcpy(d_mod, &h_mod, sizeof(modPar), cudaMemcpyHostToDevice);

	// dynamic array variables
	// list: char *file_cp;	char *file_ro;	char *file_cs;	char *file_qp;	char *file_qs;
	char *d_aux1;
	int size = strlen(h_mod.file_cp);
	cudaMalloc(&d_aux1, size*sizeof(char));
	cudaMemcpy(d_aux1, &h_mod.file_cp[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_mod[0].file_cp, &d_aux1, sizeof(char*), cudaMemcpyHostToDevice);

	char *d_aux2;
	size = strlen(h_mod.file_ro);
	cudaMalloc(&d_aux2, size*sizeof(char));
	cudaMemcpy(d_aux2, &h_mod.file_ro[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_mod[0].file_ro, &d_aux2, sizeof(char*), cudaMemcpyHostToDevice);

	if (h_mod.ischeme>2 && h_mod.ischeme!=5) {
		char *d_aux3;
		size = strlen(h_mod.file_cs);
		cudaMalloc(&d_aux3, size*sizeof(char));
		cudaMemcpy(d_aux3, &h_mod.file_cs[0], size*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_mod[0].file_cs, &d_aux3, sizeof(char*), cudaMemcpyHostToDevice);
	}

	if (h_mod.ischeme==2 || h_mod.ischeme==4) {
		char *d_aux4;
		size = strlen(h_mod.file_qp);
		cudaMalloc(&d_aux4, size*sizeof(char));
		cudaMemcpy(d_aux4, &h_mod.file_qp[0], size*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_mod[0].file_qp, &d_aux4, sizeof(char*), cudaMemcpyHostToDevice);

		char *d_aux5;
		size = strlen(h_mod.file_qs);
		cudaMalloc(&d_aux5, size*sizeof(char));
		cudaMemcpy(d_aux5, &h_mod.file_qs[0], size*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_mod[0].file_qs, &d_aux5, sizeof(char*), cudaMemcpyHostToDevice);
	}
}

void cudaFree_modPar(modPar *d_mod){
/*
Deep free's GPU structure modPar
*/
	void **d_aux;
	
	// list: char *file_cp;	char *file_ro;	char *file_cs;	char *file_qp;	char *file_qs;
	// free file_cp
	cudaMemcpy(&d_aux, &d_mod[0].file_cp, sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);
	// free file_ro
	cudaMemcpy(&d_aux, &d_mod[0].file_ro, sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	int ischeme;
	cudaMemcpy(&ischeme, &d_mod[0].ischeme, 1*sizeof(int), cudaMemcpyDeviceToHost);
	if (ischeme>2 && ischeme!=5) {
		// free file_cs
		cudaMemcpy(&d_aux, &d_mod[0].file_cs, sizeof(char*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
	}
	if (ischeme==2 || ischeme==4) {
		// free file_qp
		cudaMemcpy(&d_aux, &d_mod[0].file_qp, sizeof(char*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
		// free file_qs
		cudaMemcpy(&d_aux, &d_mod[0].file_qs, sizeof(char*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
	}

	//free single variables
	cudaFree(d_mod);
}


void cuda_deepcpy_wavPar(wavPar h_wav, srcPar h_src, wavPar *d_wav){
/*
"Deep copy" h_wav to d_wav
Obs, wav needs some dimension parameters from srcPar
*/
	//Memcpy h2d the "single" variables.
	//List:	int nsrcf; int nt;	int ns;	int nx;	float dt;	float ds;	float fmax;	int random;
	//int seed;	int nst;
	cudaMemcpy(d_wav, &h_wav, sizeof(wavPar), cudaMemcpyHostToDevice);

	//Deep copy dynamic arrays
	//List
	//	char *file_src;	size_t *nsamp;
	//wav->nsamp = (size_t *)malloc((wav->nx+1)*sizeof(size_t));

	char *d_aux1;
	int size = strlen(h_wav.file_src);
	cudaMalloc(&d_aux1, size*sizeof(char));
	cudaMemcpy(d_aux1, &h_wav.file_src[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_wav[0].file_src, &d_aux1, sizeof(char*), cudaMemcpyHostToDevice);

	if(h_wav.random){
		size_t *d_aux2;
		cudaMalloc(&d_aux2, size*sizeof(size_t));
		cudaMemcpy(d_aux2, &h_wav.nsamp[0], size*sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_wav[0].nsamp, &d_aux2, sizeof(size_t*), cudaMemcpyHostToDevice);	
	}
	
}

void cudaFree_wavPar(wavPar *d_wav){
/*
"Deep free" of a GPU wavPar structure
*/
	void **d_aux;

	// Free d_wav.file_src
	cudaMemcpy(&d_aux, &d_wav[0].file_src, sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_wav.nsamp
	int tmp;
	cudaMemcpy(&tmp, &d_wav[0].random, 1*sizeof(int),cudaMemcpyDeviceToHost);
	if(tmp){
		cudaMemcpy(&d_aux, &d_wav[0].nsamp, 1*sizeof(size_t*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
	}

	// Free single values
	cudaFree(d_wav);
}

void cuda_deepcpy_srcPar(srcPar h_src, srcPar *d_src){
	// memcpy H2D single variables
	// List: 
	// int n; int type; int orient; int single; int plane; int circle;
	// int array; int random; int multiwav; float angle;	float velo;	float amplitude;
	// float dip; float strike;	int distribution; int window; int injectionrate;
	// int sinkdepth;	int src_at_rcv; 
	cudaMemcpy(d_src, &h_src, sizeof(srcPar), cudaMemcpyHostToDevice);

	// Deep copy dynamic arrays
	// List: int *z; int *x; float *tbeg; float *tend;

	// We have to "manually" deep copy the dynamic arrays
	// deep copy h_src.z[]
	float *d_aux1;
	cudaMalloc(&d_aux1, h_src.n*sizeof(float));
	cudaMemcpy(d_aux1, &h_src.z[0], h_src.n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_src[0].z, &d_aux1, sizeof(float*), cudaMemcpyHostToDevice);

	// deep copy h_src.x[]
	float *d_aux2;
	cudaMalloc(&d_aux2, h_src.n*sizeof(float));
	cudaMemcpy(d_aux2, &h_src.x[0], h_src.n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_src[0].x, &d_aux2, sizeof(float*), cudaMemcpyHostToDevice);


	// deep copy h_src.tbeg[]
	float *d_aux3;
	cudaMalloc(&d_aux3, h_src.n*sizeof(float));
	cudaMemcpy(d_aux3, &h_src.tbeg[0], h_src.n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_src[0].tbeg, &d_aux3, sizeof(float*), cudaMemcpyHostToDevice);

	// deep copy h_src.tend[]
	float *d_aux4; // need extra ptr to alloc more gpu memory
	cudaMalloc(&d_aux4, h_src.n*sizeof(float));
	cudaMemcpy(d_aux4, &h_src.tend[0], h_src.n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_src[0].tend, &d_aux4, sizeof(float*), cudaMemcpyHostToDevice);

	//DO NOT free d_aux1, d_aux2, d_aux3, d_aux4
	//They contain the link to the gpu deep copy
	//They are properly freed by cudaFree_srcPar
}

void cudaFree_srcPar(srcPar *d_src){
/*
"Deepfree" gpu variable d_src, of specific structure srcPar
*/	
	void **d_aux;

	// Free d_src[0].z
	cudaMemcpy(&d_aux, &d_src[0].z, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_src[0].x
	cudaMemcpy(&d_aux, &d_src[0].x, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_src[0].tbeg
	cudaMemcpy(&d_aux, &d_src[0].tbeg, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_src[0].tend
	cudaMemcpy(&d_aux, &d_src[0].tend, sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux); //can use the same ptr to free

	// Free single values
	cudaFree(d_src);

}

void cuda_deepcpy_recPar(recPar h_rec, recPar *d_rec){
/*
"Deep copy" h_rec to d_rec
*/
	//Memcpy h2d the "single" variables.
	cudaMemcpy(d_rec, &h_rec, sizeof(recPar), cudaMemcpyHostToDevice);

	//Deep copy dynamic arrays
	//List	char *file_rcv;	int *z;	int *x;	float *zr;	float *xr;

	char *d_aux1;
	int size = strlen(h_rec.file_rcv);
	cudaMalloc(&d_aux1, size*sizeof(char));
	cudaMemcpy(d_aux1, &h_rec.file_rcv[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_rec[0].file_rcv, &d_aux1, sizeof(char*), cudaMemcpyHostToDevice);

	// copy h_rec.z
	int *d_aux2;
	size = h_rec.max_nrec;
	cudaMalloc(&d_aux2, size*sizeof(int));
	cudaMemcpy(d_aux2, &h_rec.z[0], size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_rec[0].z, &d_aux2, 1*sizeof(int*), cudaMemcpyHostToDevice);

	// copy h_rec.x
	int *d_aux3;
	size = h_rec.max_nrec;
	cudaMalloc(&d_aux3, size*sizeof(int));
	cudaMemcpy(d_aux3, &h_rec.x[0], size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_rec[0].x, &d_aux3, 1*sizeof(int*), cudaMemcpyHostToDevice);

	// copy h_rec.zr
	float *d_aux4;
	size = h_rec.max_nrec;
	cudaMalloc(&d_aux4, size*sizeof(float));
	cudaMemcpy(d_aux4, &h_rec.zr[0], size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_rec[0].zr, &d_aux4, 1*sizeof(float*), cudaMemcpyHostToDevice);

	// copy h_rec.xr
	float *d_aux5;
	size = h_rec.max_nrec;
	cudaMalloc(&d_aux5, size*sizeof(float));
	cudaMemcpy(d_aux5, &h_rec.xr[0], size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_rec[0].xr, &d_aux5, 1*sizeof(float*), cudaMemcpyHostToDevice);

}

void cudaFree_recPar(recPar *d_rec){
	void **d_aux;

	// Free d_rec[0].file_rcv
	cudaMemcpy(&d_aux, &d_rec[0].file_rcv, 1*sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_rec[0].z
	cudaMemcpy(&d_aux, &d_rec[0].z, 1*sizeof(int*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_rec[0].x
	cudaMemcpy(&d_aux, &d_rec[0].x, 1*sizeof(int*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_rec[0].zr
	cudaMemcpy(&d_aux, &d_rec[0].zr, 1*sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free d_rec[0].xr
	cudaMemcpy(&d_aux, &d_rec[0].xr, 1*sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// Free single variables
	cudaFree(d_rec);
}

void cuda_deepcpy_snaPar(snaPar h_sna, snaPar *d_sna){
/*
Deep copies to GPU h_sna -> d_sna
*/
	// Copy single variables
	// List: compType type;	int nsnap; int delay; int skipdt; int skipdz; int skipdx;	
	//       int nz; int nx; int z1; int z2; int x1; int x2; int vxvztime; int beam; int withbnd;
	cudaMemcpy(d_sna, &h_sna, sizeof(snaPar), cudaMemcpyHostToDevice);

	// Deep copy dynamic arrays
	// List: char *file_snap;	char *file_beam;
	char *d_aux1;
	int size = strlen(h_sna.file_snap);
	cudaMalloc(&d_aux1, size*sizeof(char));
	cudaMemcpy(d_aux1, &h_sna.file_snap[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_sna[0].file_snap, &d_aux1, 1*sizeof(char*), cudaMemcpyHostToDevice);

	char *d_aux2;
	size = strlen(h_sna.file_beam);
	cudaMalloc(&d_aux2, size*sizeof(char));
	cudaMemcpy(d_aux2, &h_sna.file_beam[0], size*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_sna[0].file_beam, &d_aux2, 1*sizeof(char*), cudaMemcpyHostToDevice);
}

void cudaFree_snaPar(snaPar *d_sna){
	void **d_aux;

	// deep free d_sna[0].file_snap
	cudaMemcpy(&d_aux, &d_sna[0].file_snap, 1*sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// deep free d_sna[0].file_beam
	cudaMemcpy(&d_aux, &d_sna[0].file_beam, 1*sizeof(char*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// free single variables
	cudaFree(d_sna);
}

void cuda_deepcpy_shotPar(shotPar h_shot, shotPar *d_shot){
	// single variables
	// List: 	int n;
	cudaMemcpy(d_shot, &h_shot, 1*sizeof(shotPar), cudaMemcpyHostToDevice);

	// dynamic arrays
	// List:	int *z;	int *x;
	int *d_aux1;
	int size = h_shot.n;
	cudaMalloc(&d_aux1, size*sizeof(int));
	cudaMemcpy(d_aux1, &h_shot.z[0], size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_shot[0].z, &d_aux1, 1*sizeof(int*), cudaMemcpyHostToDevice);

	int *d_aux2;
	size = h_shot.n;
	cudaMalloc(&d_aux2, size*sizeof(int));
	cudaMemcpy(d_aux2, &h_shot.x[0], size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_shot[0].x, &d_aux2, 1*sizeof(int*), cudaMemcpyHostToDevice);
}

void cudaFree_shotPar(shotPar *d_shot){
	void **d_aux;

	// free d_shot[0].z
	cudaMemcpy(&d_aux, &d_shot[0].z, 1*sizeof(int*), cudaMemcpyDeviceToHost);	
	cudaFree(d_aux);

	// free d_shot[0].x
	cudaMemcpy(&d_aux, &d_shot[0].x, 1*sizeof(int*), cudaMemcpyDeviceToHost);	
	cudaFree(d_aux);

	// free single vars
	cudaFree(d_shot);
}

void cuda_deepcpy_bndPar(bndPar h_bnd, modPar h_mod, bndPar *d_bnd){
/*
Deep copies h_bnd -> d_bnd
Obs, bnd needs some dimension parameters from modPar
*/
	int size;

	// Copy single variables
	// List	int top;int bot;int lef;int rig;int cfree;int ntap;int npml;float R;float m;
	cudaMemcpy(d_bnd, &h_bnd, sizeof(bndPar), cudaMemcpyHostToDevice);

	// Deep copy dynamic arrays
	// List: float *tapz; float *tapx; float *tapxz; int *surface; float *pml_Vx; 
	//       float *pml_nzVx; float *pml_nxVz; float *pml_nzVz; 
	//       float *pml_nxP; float *pml_nzP;
	

	if(h_bnd.ntap){
		//tapz
		float *d_aux1;
	    size = h_bnd.ntap;
	    cudaMalloc(&d_aux1, size*sizeof(float));
	    cudaMemcpy(d_aux1, &h_bnd.tapz[0], size*sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(&d_bnd[0].tapz, &d_aux1, 1*sizeof(float*), cudaMemcpyHostToDevice);

	    //tapx
	   	float *d_aux2;
	    size = h_bnd.ntap;
	    cudaMalloc(&d_aux2, size*sizeof(float));
	    cudaMemcpy(d_aux2, &h_bnd.tapx[0], size*sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(&d_bnd[0].tapx, &d_aux2, 1*sizeof(float*), cudaMemcpyHostToDevice);

		// tapxz
		float *d_aux3;
	    size = h_bnd.ntap*h_bnd.ntap;
	    cudaMalloc(&d_aux3, size*sizeof(float));
	    cudaMemcpy(d_aux3, &h_bnd.tapxz[0], size*sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(&d_bnd[0].tapxz, &d_aux3, 1*sizeof(float*), cudaMemcpyHostToDevice);
	}

	// surface    
	int *d_aux4;
	size = h_mod.nax+h_mod.naz;
	cudaMalloc(&d_aux4, size*sizeof(int));
	cudaMemcpy(d_aux4, &h_bnd.surface[0], size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_bnd[0].surface, &d_aux4, 1*sizeof(int*), cudaMemcpyHostToDevice);

	// Obs, *pml_Vx, *pml_nzVx, *pml_nxVz, *pml_nzVz; *pml_nxP; *pml_nzP;
	// are "dead" variables (only exist at the struct definition, but are not used 
	// anywhere in the current version of fdelmodc)
}

void cudaFree_bndPar(bndPar *d_bnd){
/*
Deep frees GPU structure bndPar
*/	
	void **d_aux;

	int key;
	cudaMemcpy(&key, &d_bnd[0].ntap, 1*sizeof(int), cudaMemcpyDeviceToHost);
	if(key){
		//tapz
		cudaMemcpy(&d_aux, &d_bnd[0].tapz, 1*sizeof(float*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
	    //tapx
	    cudaMemcpy(&d_aux, &d_bnd[0].tapx, 1*sizeof(float*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
		// tapxz
		cudaMemcpy(&d_aux, &d_bnd[0].tapxz, 1*sizeof(float*), cudaMemcpyDeviceToHost);
		cudaFree(d_aux);
	}

	//surface
	cudaMemcpy(&d_aux, &d_bnd[0].surface, 1*sizeof(int*), cudaMemcpyDeviceToHost);
	cudaFree(d_aux);

	// free single variables
	cudaFree(d_bnd);
}




// PRINT FUNCTIONS
// AUXILIARY FDELMODC FUNCTIONS
void printf_modPar(modPar mod){
		printf("printf_modPar.\n");
		printf("mod.iorder=%d\n", mod.iorder);
		printf("mod.ischeme=%d\n", mod.ischeme);
		printf("mod.grid_dir=%d\n", mod.grid_dir);
		printf("mod.sh=%d\n", mod.sh);
		printf("mod.file_cp=%s\n", mod.file_cp);
		printf("mod.file_ro=%s\n", mod.file_ro);
		if(mod.ischeme>2 && mod.ischeme!=5) printf("mod.file_cs=%s\n", mod.file_cs);
		if(mod.ischeme==2 || mod.ischeme==4) printf("mod.file_qp=%s\n", mod.file_qp);
		if(mod.ischeme==2 || mod.ischeme==4) printf("mod.file_qs=%s\n", mod.file_qs);
		printf("mod.dz=%f\n", mod.dz);
		printf("mod.dx=%f\n", mod.dx);
		printf("mod.dt=%f\n", mod.dt);
		printf("mod.tmod=%f\n", mod.tmod);
		printf("mod.nt=%d\n", mod.nt);
		printf("mod.z0=%f\n", mod.z0);
		printf("mod.x0=%f\n", mod.x0);
		printf("mod.cp_min=%f\n", mod.cp_min);
		printf("mod.cp_max=%f\n", mod.cp_max);
		printf("mod.cs_min=%f\n", mod.cs_min);
		printf("mod.cs_max=%f\n", mod.cs_max);
		printf("mod.ro_min=%f\n", mod.ro_min);
		printf("mod.ro_max=%f\n", mod.ro_max);
		printf("mod.nz=%d\n", mod.nz);
		printf("mod.nx=%d\n", mod.nx);
		printf("mod.naz=%d\n", mod.naz);
		printf("mod.nax=%d\n", mod.nax);
		printf("mod.ioXx=%d\n", mod.ioXx);
		printf("mod.ioXz=%d\n", mod.ioXz);
		printf("mod.ieXx=%d\n", mod.ieXx);
		printf("mod.ieXz=%d\n", mod.ieXz);
		printf("mod.ioZx=%d\n", mod.ioZx);
		printf("mod.ioZz=%d\n", mod.ioZz);
		printf("mod.ieZx=%d\n", mod.ieZx);
		printf("mod.ieZz=%d\n", mod.ieZz);
		printf("mod.ioPx=%d\n", mod.ioPx);
		printf("mod.ioPz=%d\n", mod.ioPz);
		printf("mod.iePx=%d\n", mod.iePx);
		printf("mod.iePz=%d\n", mod.iePz);
		printf("mod.ioTx=%d\n", mod.ioTx);
		printf("mod.ioTz=%d\n", mod.ioTz);
		printf("mod.ieTx=%d\n", mod.ieTx);
		printf("mod.ieTz=%d\n", mod.ieTz);
		printf("mod.Qp=%f\n", mod.Qp);
		printf("mod.Qs=%f\n", mod.Qs);
		printf("mod.fw=%f\n", mod.fw);
		printf("mod.qr=%f\n", mod.qr);
}

__global__ void kernel_printf_modPar(modPar *d_mod){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	printf("kernel_printf_modPar. Hello from thread %d.\n", id);

	printf("d_mod[0].iorder=%d\n", d_mod[0].iorder);
	printf("d_mod[0].ischeme=%d\n", d_mod[0].ischeme);
	printf("d_mod[0].grid_dir=%d\n", d_mod[0].grid_dir);
	printf("d_mod[0].sh=%d\n", d_mod[0].sh);
	printf("d_mod[0].file_cp=%s\n", d_mod[0].file_cp);
	printf("d_mod[0].file_ro=%s\n", d_mod[0].file_ro);
	if(d_mod[0].ischeme>2 && d_mod[0].ischeme!=5) printf("d_mod[0].file_cs=%s\n", d_mod[0].file_cs);
	if(d_mod[0].ischeme==2 || d_mod[0].ischeme==4) printf("d_mod[0].file_qp=%s\n", d_mod[0].file_qp);
	if(d_mod[0].ischeme==2 || d_mod[0].ischeme==4) printf("d_mod[0].file_qs=%s\n", d_mod[0].file_qs);
	printf("d_mod[0].dz=%f\n", d_mod[0].dz);
	printf("d_mod[0].dx=%f\n", d_mod[0].dx);
	printf("d_mod[0].dt=%f\n", d_mod[0].dt);
	printf("d_mod[0].tmod[0]=%f\n", d_mod[0].tmod);
	printf("d_mod[0].nt=%d\n", d_mod[0].nt);
	printf("d_mod[0].z0=%f\n", d_mod[0].z0);
	printf("d_mod[0].x0=%f\n", d_mod[0].x0);
	printf("d_mod[0].cp_min=%f\n", d_mod[0].cp_min);
	printf("d_mod[0].cp_max=%f\n", d_mod[0].cp_max);
	printf("d_mod[0].cs_min=%f\n", d_mod[0].cs_min);
	printf("d_mod[0].cs_max=%f\n", d_mod[0].cs_max);
	printf("d_mod[0].ro_min=%f\n", d_mod[0].ro_min);
	printf("d_mod[0].ro_max=%f\n", d_mod[0].ro_max);
	printf("d_mod[0].nz=%d\n", d_mod[0].nz);
	printf("d_mod[0].nx=%d\n", d_mod[0].nx);
	printf("d_mod[0].naz=%d\n", d_mod[0].naz);
	printf("d_mod[0].nax=%d\n", d_mod[0].nax);
	printf("d_mod[0].ioXx=%d\n", d_mod[0].ioXx);
	printf("d_mod[0].ioXz=%d\n", d_mod[0].ioXz);
	printf("d_mod[0].ieXx=%d\n", d_mod[0].ieXx);
	printf("d_mod[0].ieXz=%d\n", d_mod[0].ieXz);
	printf("d_mod[0].ioZx=%d\n", d_mod[0].ioZx);
	printf("d_mod[0].ioZz=%d\n", d_mod[0].ioZz);
	printf("d_mod[0].ieZx=%d\n", d_mod[0].ieZx);
	printf("d_mod[0].ieZz=%d\n", d_mod[0].ieZz);
	printf("d_mod[0].ioPx=%d\n", d_mod[0].ioPx);
	printf("d_mod[0].ioPz=%d\n", d_mod[0].ioPz);
	printf("d_mod[0].iePx=%d\n", d_mod[0].iePx);
	printf("d_mod[0].iePz=%d\n", d_mod[0].iePz);
	printf("d_mod[0].ioTx=%d\n", d_mod[0].ioTx);
	printf("d_mod[0].ioTz=%d\n", d_mod[0].ioTz);
	printf("d_mod[0].ieTx=%d\n", d_mod[0].ieTx);
	printf("d_mod[0].ieTz=%d\n", d_mod[0].ieTz);
	printf("d_mod[0].Qp=%f\n", d_mod[0].Qp);
	printf("d_mod[0].Qs=%f\n", d_mod[0].Qs);
	printf("d_mod[0].fw=%f\n", d_mod[0].fw);
	printf("d_mod[0].qr=%f\n", d_mod[0].qr);
}

void printf_wavPar(wavPar wav){
	printf("printf_wavPar. \n");
    printf("wav.file_src=%s\n", wav.file_src);
    printf("wav.nsrcf=%d\n", wav.nsrcf);
    printf("wav.nt=%d\n", wav.nt);
    printf("wav.ns=%d\n", wav.ns);
    printf("wav.nx=%d\n", wav.nx);
    printf("wav.dt=%f\n", wav.dt);
    printf("wav.ds=%f\n", wav.ds);
    printf("wav.fmax=%f\n", wav.fmax);
    printf("wav.random=%d\n", wav.random);
    printf("wav.seed=%d\n", wav.seed);
    printf("wav.nst=%d\n", wav.nst);
    printf("wav.nsamp[0]=%li\n", wav.nsamp[0]);
}

__global__ void kernel_printf_wavPar(wavPar *d_wav){
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	printf("kernel_printf_wavPar. Hello from thread %d\n", id);

	printf("d_wav[0].file_src=%s\n", d_wav[0].file_src);
    printf("d_wav[0].nsrcf=%d\n", d_wav[0].nsrcf);
    printf("d_wav[0].nt=%d\n", d_wav[0].nt);
    printf("d_wav[0].ns=%d\n", d_wav[0].ns);
    printf("d_wav[0].nx=%d\n", d_wav[0].nx);
    printf("d_wav[0].dt=%f\n", d_wav[0].dt);
    printf("d_wav[0].ds=%f\n", d_wav[0].ds);
    printf("d_wav[0].fmax=%f\n", d_wav[0].fmax);
    printf("d_wav[0].random=%d\n", d_wav[0].random);
    printf("d_wav[0].seed=%d\n", d_wav[0].seed);
    printf("d_wav[0].nst=%d\n", d_wav[0].nst);
    printf("d_wav[0].nsamp[0]=%li\n", d_wav[0].nsamp[0]);	
}


void printf_srcPar(srcPar src){
	printf("printf_srcPar\n");

	printf("src.n=%d\n", src.n);
	printf("src.type=%d\n", src.type);
	printf("src.orient=%d\n", src.orient);
	//printf("src.z=%p\n", src.z);
	printf("src.z[0]=%d\n", src.z[0]);
	//printf("src.x=%p\n", src.x);
	printf("src.x[0]=%d\n", src.x[0]);
	printf("src.single=%d\n", src.single);
	printf("src.plane=%d\n", src.plane);
	printf("src.circle=%d\n", src.circle);
	printf("src.array=%d\n", src.array);
	printf("src.random=%d\n", src.random);
	//printf("src.tbeg=%p\n", src.tbeg);
	printf("src.tbeg[0]=%f\n", src.tbeg[0]);
	//printf("src.tend=%p\n", src.tend);
	printf("src.tend[0]=%f\n", src.tend[0]);
	printf("src.multiwav=%d\n", src.multiwav);
	printf("src.angle=%f\n", src.angle);
	printf("src.velo=%f\n", src.velo);
	printf("src.amplitude=%f\n", src.amplitude);
	printf("src.dip=%f\n", src.dip);
	printf("src.strike=%f\n", src.strike);
	printf("src.distribution=%d\n", src.distribution);
	printf("src.window=%d\n", src.window);
	printf("src.injectionrate=%d\n", src.injectionrate);
	printf("src.sinkdepth=%d\n", src.sinkdepth);
	printf("src.src_at_rcv=%d\n", src.src_at_rcv);
}

__global__ void kernel_printf_srcPar(srcPar *d_src){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	printf("kernel_print_srcPar. I am thread %d.\n", id);

	printf("d_src[0].n=%d\n", d_src[0].n);
	printf("d_src[0].type=%d\n", d_src[0].type);
	printf("d_src[0].orient=%d\n", d_src[0].orient);
	//printf("d_src[0].z=%p\n", d_src[0].z);
	printf("d_src[0].z[0]=%f\n", d_src[0].z[0]);
	// printf("d_src[0].x=%p\n", d_src[0].x);
	printf("d_src[0].x[0]=%f\n", d_src[0].x[0]);
	printf("d_src[0].single=%d\n", d_src[0].single);
	printf("d_src[0].plane=%d\n", d_src[0].plane);
	printf("d_src[0].circle=%d\n", d_src[0].circle);
	printf("d_src[0].array=%d\n", d_src[0].array);
	printf("d_src[0].random=%d\n", d_src[0].random);
	// printf("d_src[0].tbeg=%p\n", d_src[0].tbeg);
	printf("d_src[0].tbeg[0]=%f\n", d_src[0].tbeg[0]);
	//printf("d_src[0].tend=%p\n", d_src[0].tend);
	printf("d_src[0].tend[0]=%f\n", d_src[0].tend[0]);
	printf("d_src[0].multiwav=%d\n", d_src[0].multiwav);
	printf("d_src[0].angle=%f\n", d_src[0].angle);
	printf("d_src[0].velo=%f\n", d_src[0].velo);
	printf("d_src[0].amplitude=%f\n", d_src[0].amplitude);
	printf("d_src[0].dip=%f\n", d_src[0].dip);
	printf("d_src[0].strike=%f\n", d_src[0].strike);
	printf("d_src[0].distribution=%d\n", d_src[0].distribution);
	printf("d_src[0].window=%d\n", d_src[0].window);
	printf("d_src[0].injectionrate=%d\n", d_src[0].injectionrate);
	printf("d_src[0].sinkdepth=%d\n", d_src[0].sinkdepth);
	printf("d_src[0].src_at_rcv=%d\n", d_src[0].src_at_rcv);
}

__global__ void kernel_printf_srcPar2(srcPar *d_src){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	printf("kernel_print_srcPar. I am thread %d.\n", id);

	printf("d_src->n=%d\n", d_src->n);
	printf("d_src->type=%d\n", d_src->type);
	printf("d_src->orient=%d\n", d_src->orient);
	//printf("d_src->z=%p\n", d_src->z);
	printf("d_src->z[0]=%f\n", d_src->z[0]);
	// printf("d_src->x=%p\n", d_src->x);
	printf("d_src->x[0]=%f\n", d_src->x[0]);
	printf("d_src->single=%d\n", d_src->single);
	printf("d_src->plane=%d\n", d_src->plane);
	printf("d_src->circle=%d\n", d_src->circle);
	printf("d_src->array=%d\n", d_src->array);
	printf("d_src->random=%d\n", d_src->random);
	// printf("d_src->tbeg=%p\n", d_src->tbeg);
	printf("d_src->tbeg[0]=%f\n", d_src->tbeg[0]);
	//printf("d_src->tend=%p\n", d_src->tend);
	printf("d_src->tend[0]=%f\n", d_src->tend[0]);
	printf("d_src->multiwav=%d\n", d_src->multiwav);
	printf("d_src->angle=%f\n", d_src->angle);
	printf("d_src->velo=%f\n", d_src->velo);
	printf("d_src->amplitude=%f\n", d_src->amplitude);
	printf("d_src->dip=%f\n", d_src->dip);
	printf("d_src->strike=%f\n", d_src->strike);
	printf("d_src->distribution=%d\n", d_src->distribution);
	printf("d_src->window=%d\n", d_src->window);
	printf("d_src->injectionrate=%d\n", d_src->injectionrate);
	printf("d_src->sinkdepth=%d\n", d_src->sinkdepth);
	printf("d_src->src_at_rcv=%d\n", d_src->src_at_rcv);
}

void printf_recPar(recPar rec){
	printf("printf_recPar.\n");

	printf("rec.file_rcv=%s\n", rec.file_rcv);
	printf("\trec.type.vz=%d\n", rec.type.vz);
	printf("\trec.type.vx=%d\n", rec.type.vx);
	printf("\trec.type.p=%d\n", rec.type.p);
	printf("\trec.type.txx=%d\n", rec.type.txx);
	printf("\trec.type.tzz=%d\n", rec.type.tzz);
	printf("\trec.type.txz=%d\n", rec.type.txz);
	printf("\trec.type.pp=%d\n", rec.type.pp);
	printf("\trec.type.ss=%d\n", rec.type.ss);
	printf("\trec.type.ud=%d\n", rec.type.ud);
	printf("rec.n=%d\n", rec.n);
	printf("rec.nt=%d\n", rec.nt);
	printf("rec.delay=%d\n", rec.delay);
	printf("rec.skipdt=%d\n", rec.skipdt);
	printf("rec.max_nrec=%d\n", rec.max_nrec);
	printf("rec.z[0]=%d\n", rec.z[0]);
	printf("rec.x[0]=%d\n", rec.x[0]);
	printf("rec.zr[0]=%f\n", rec.zr[0]);
	printf("rec.xr[0]=%f\n", rec.xr[0]);
	printf("rec.int_p=%d\n", rec.int_p);
	printf("rec.int_vx=%d\n", rec.int_vx);
	printf("rec.int_vz=%d\n", rec.int_vz);
	printf("rec.scale=%d\n", rec.scale);
	printf("rec.sinkdepth=%d\n", rec.sinkdepth);
	printf("rec.sinkvel=%d\n", rec.sinkvel);
	printf("rec.cp=%f\n", rec.cp);
	printf("rec.rho=%f\n", rec.rho);
}

__global__ void kernel_printf_recPar(recPar *d_rec){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	printf("kernel_print_recPar. I am thread %d.\n", id);

	printf("d_rec[0].file_rcv=%s\n", d_rec[0].file_rcv);
	printf("\td_rec[0].type.vz=%d\n", d_rec[0].type.vz);
	printf("\td_rec[0].type.vx=%d\n", d_rec[0].type.vx);
	printf("\td_rec[0].type.p=%d\n", d_rec[0].type.p);
	printf("\td_rec[0].type.txx=%d\n", d_rec[0].type.txx);
	printf("\td_rec[0].type.tzz=%d\n", d_rec[0].type.tzz);
	printf("\td_rec[0].type.txz=%d\n", d_rec[0].type.txz);
	printf("\td_rec[0].type.pp=%d\n", d_rec[0].type.pp);
	printf("\td_rec[0].type.ss=%d\n", d_rec[0].type.ss);
	printf("\td_rec[0].type.ud=%d\n", d_rec[0].type.ud);
	printf("d_rec[0].n=%d\n", d_rec[0].n);
	printf("d_rec[0].nt=%d\n", d_rec[0].nt);
	printf("d_rec[0].delay=%d\n", d_rec[0].delay);
	printf("d_rec[0].skipdt=%d\n", d_rec[0].skipdt);
	printf("d_rec[0].max_nrec=%d\n", d_rec[0].max_nrec);
	printf("d_rec[0].z[0]=%d\n", d_rec[0].z[0]);
	printf("d_rec[0].x[0]=%d\n", d_rec[0].x[0]);
	printf("d_rec[0].zr[0]=%f\n", d_rec[0].zr[0]);
	printf("d_rec[0].xr[0]=%f\n", d_rec[0].xr[0]);
	printf("d_rec[0].int_p=%d\n", d_rec[0].int_p);
	printf("d_rec[0].int_vx=%d\n", d_rec[0].int_vx);
	printf("d_rec[0].int_vz=%d\n", d_rec[0].int_vz);
	printf("d_rec[0].scale=%d\n", d_rec[0].scale);
	printf("d_rec[0].sinkdepth=%d\n", d_rec[0].sinkdepth);
	printf("d_rec[0].sinkvel=%d\n", d_rec[0].sinkvel);
	printf("d_rec[0].cp=%f\n", d_rec[0].cp);
	printf("d_rec[0].rho=%f\n", d_rec[0].rho);
}


void printf_snaPar(snaPar sna){
	printf("printf_snaPar.\n");
	printf("sna.file_snap=%s\n", sna.file_snap);
	printf("sna.file_beam=%s\n", sna.file_beam);
	printf("sna.type.vz=%d\n", sna.type.vz);
	printf("sna.type.vx=%d\n", sna.type.vx);
	printf("sna.type.p=%d\n", sna.type.p);
	printf("sna.type.txx=%d\n", sna.type.txx);
	printf("sna.type.tzz=%d\n", sna.type.tzz);
	printf("sna.type.txz=%d\n", sna.type.txz);
	printf("sna.type.pp=%d\n", sna.type.pp);
	printf("sna.type.ss=%d\n", sna.type.ss);
	printf("sna.type.ud=%d\n", sna.type.ud);
	printf("sna.nsnap=%d\n", sna.nsnap);
	printf("sna.delay=%d\n", sna.delay);
	printf("sna.skipdt=%d\n", sna.skipdt);
	printf("sna.skipdz=%d\n", sna.skipdz);
	printf("sna.skipdx=%d\n", sna.skipdx);
	printf("sna.nz=%d\n", sna.nz);
	printf("sna.nx=%d\n", sna.nx);
	printf("sna.z1=%d\n", sna.z1);
	printf("sna.z2=%d\n", sna.z2);
	printf("sna.x1=%d\n", sna.x1);
	printf("sna.x2=%d\n", sna.x2);
	printf("sna.vxvztime=%d\n", sna.vxvztime);
	printf("sna.beam=%d\n", sna.beam);
	printf("sna.withbnd=%d\n", sna.withbnd);
}

__global__ void kernel_printf_snaPar(snaPar *d_sna){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	printf("kernel_print_recPar. I am thread %d.\n", id);

	printf("d_sna[0].file_snap=%s\n", d_sna[0].file_snap);
	printf("d_sna[0].file_beam=%s\n", d_sna[0].file_beam);
	printf("d_sna[0].type.vz=%d\n", d_sna[0].type.vz);
	printf("d_sna[0].type.vx=%d\n", d_sna[0].type.vx);
	printf("d_sna[0].type.p=%d\n", d_sna[0].type.p);
	printf("d_sna[0].type.txx=%d\n", d_sna[0].type.txx);
	printf("d_sna[0].type.tzz=%d\n", d_sna[0].type.tzz);
	printf("d_sna[0].type.txz=%d\n", d_sna[0].type.txz);
	printf("d_sna[0].type.pp=%d\n", d_sna[0].type.pp);
	printf("d_sna[0].type.ss=%d\n", d_sna[0].type.ss);
	printf("d_sna[0].type.ud=%d\n", d_sna[0].type.ud);
	printf("d_sna[0].nsnap=%d\n", d_sna[0].nsnap);
	printf("d_sna[0].delay=%d\n", d_sna[0].delay);
	printf("d_sna[0].skipdt=%d\n", d_sna[0].skipdt);
	printf("d_sna[0].skipdz=%d\n", d_sna[0].skipdz);
	printf("d_sna[0].skipdx=%d\n", d_sna[0].skipdx);
	printf("d_sna[0].nz=%d\n", d_sna[0].nz);
	printf("d_sna[0].nx=%d\n", d_sna[0].nx);
	printf("d_sna[0].z1=%d\n", d_sna[0].z1);
	printf("d_sna[0].z2=%d\n", d_sna[0].z2);
	printf("d_sna[0].x1=%d\n", d_sna[0].x1);
	printf("d_sna[0].x2=%d\n", d_sna[0].x2);
	printf("d_sna[0].vxvztime=%d\n", d_sna[0].vxvztime);
	printf("d_sna[0].beam=%d\n", d_sna[0].beam);
	printf("d_sna[0].withbnd=%d\n", d_sna[0].withbnd);
}


void printf_shotPar(shotPar shot){
	printf("printf_shotPar.\n");	
	
	printf("shot.n=%d\n", shot.n);
	printf("shot.z[0]=%d\n", shot.z[0]);
	printf("shot.x[0]=%d\n", shot.x[0]);
}

__global__ void kernel_printf_shotPar(shotPar *d_shot){
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	printf("kernel_print_shotPar. I am thread %d.\n", id);

	printf("d_shot[0].n=%d\n", d_shot[0].n);
	printf("d_shot[0].z[0]=%d\n", d_shot[0].z[0]);
	printf("d_shot[0].x[0]=%d\n", d_shot[0].x[0]);
}

void printf_bndPar(bndPar bnd){
	printf("printf_bndPar.\n");	

	printf("bnd.top=%d\n", bnd.top);
	printf("bnd.bot=%d\n", bnd.bot);
	printf("bnd.lef=%d\n", bnd.lef);
	printf("bnd.rig=%d\n", bnd.rig);
	printf("bnd.tapz[0]=%f\n", bnd.tapz[0]);
	printf("bnd.tapz[1]=%f\n", bnd.tapz[1]);
	printf("bnd.tapz[n-2]=%f\n", bnd.tapz[bnd.ntap-2]);
	printf("bnd.tapz[n-1]=%f\n", bnd.tapz[bnd.ntap-1]);
	printf("bnd.tapx[0]=%f\n", bnd.tapx[0]);
	printf("bnd.tapx[1]=%f\n", bnd.tapx[1]);
	printf("bnd.tapx[n-2]=%f\n", bnd.tapx[bnd.ntap-2]);
	printf("bnd.tapx[n-1]=%f\n", bnd.tapx[bnd.ntap-1]);
	printf("bnd.tapxz[0]=%f\n", bnd.tapxz[0]);
	printf("bnd.tapxz[1]=%f\n", bnd.tapxz[1]);
	printf("bnd.tapxz[n-2]=%f\n", bnd.tapxz[bnd.ntap-2]);
	printf("bnd.tapxz[n-1]=%f\n", bnd.tapxz[bnd.ntap-1]);
	printf("bnd.cfree=%d\n", bnd.cfree);
	printf("bnd.ntap=%d\n", bnd.ntap);
	printf("bnd.surface[0]=%d\n", bnd.surface[0]);
	printf("bnd.npml=%d\n", bnd.npml);
	printf("bnd.R=%f\n", bnd.R);
	printf("bnd.m=%f\n", bnd.m);

	printf("Obs, *pml_Vx; *pml_nzVx;*pml_nxVz; *pml_nzVz;*pml_nxP;*pml_nzP;\
are 'dead' variables in the current version of fdelmodc.\n");
}

__global__ void kernel_printf_bndPar(bndPar *d_bnd){
	printf("printf_bndPar.\n");	

	printf("d_bnd[0].top=%d\n", d_bnd[0].top);
	printf("d_bnd[0].bot=%d\n", d_bnd[0].bot);
	printf("d_bnd[0].lef=%d\n", d_bnd[0].lef);
	printf("d_bnd[0].rig=%d\n", d_bnd[0].rig);
	printf("d_bnd[0].tapz[0]=%f\n", d_bnd[0].tapz[0]);
	printf("d_bnd[0].tapz[1]=%f\n", d_bnd[0].tapz[1]);
	printf("d_bnd[0].tapz[n-2]=%f\n", d_bnd[0].tapz[d_bnd[0].ntap-2]);
	printf("d_bnd[0].tapz[n-1]=%f\n", d_bnd[0].tapz[d_bnd[0].ntap-1]);
	printf("d_bnd[0].tapx[0]=%f\n", d_bnd[0].tapx[0]);
	printf("d_bnd[0].tapx[1]=%f\n", d_bnd[0].tapx[1]);
	printf("d_bnd[0].tapx[n-2]=%f\n", d_bnd[0].tapx[d_bnd[0].ntap-2]);
	printf("d_bnd[0].tapx[n-1]=%f\n", d_bnd[0].tapx[d_bnd[0].ntap-1]);
	printf("d_bnd[0].tapxz[0]=%f\n", d_bnd[0].tapx[0]);
	printf("d_bnd[0].tapxz[1]=%f\n", d_bnd[0].tapx[1]);
	printf("d_bnd[0].tapxz[n-2]=%f\n", d_bnd[0].tapxz[d_bnd[0].ntap-2]);
	printf("d_bnd[0].tapxz[n-1]=%f\n", d_bnd[0].tapxz[d_bnd[0].ntap-1]);
	printf("d_bnd[0].cfree=%d\n", d_bnd[0].cfree);
	printf("d_bnd[0].ntap=%d\n", d_bnd[0].ntap);
	printf("d_bnd[0].surface[0]=%d\n", d_bnd[0].surface[0]);
	printf("d_bnd[0].npml=%d\n", d_bnd[0].npml);
	printf("d_bnd[0].R=%f\n", d_bnd[0].R);
	printf("d_bnd[0].m=%f\n", d_bnd[0].m);

	printf("Obs, *pml_Vx; *pml_nzVx;*pml_nxVz; *pml_nzVz;*pml_nxP;*pml_nzP;\
are 'dead' variables in the current version of fdelmodc.\n");
}




