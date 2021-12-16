#ifndef _CUDA_FDEL2D_PARAMETERS_CUH
#define _CUDA_FDEL2D_PARAMETERS_CUH

#include "fdelmodc.h" 

void cuda_printWhichGPU(char *text);
void cuda_initParStructs(modPar mod, srcPar src, wavPar wav, bndPar bnd, 
					recPar rec, shotPar shot, snaPar sna,
					modPar **d_mod, srcPar **d_src, wavPar **d_wav, bndPar **d_bnd, 
					recPar **d_rec, shotPar **d_shot, snaPar **d_sna, int verbose);
void cuda_freeParStructs(modPar *d_mod, srcPar *d_src, wavPar *d_wav, bndPar *d_bnd, 
						 recPar *d_rec, shotPar *d_shot, snaPar *d_sna, int verbose);
void cuda_initCoefFields(modPar mod, float *rox, float *roz, float *l2m, float *lam, float *mul,
						float *tss, float *tes, float *tep, float *p, float *q, float *r,
						float **d_rox, float **d_roz, float **d_l2m, float **d_lam, float **d_mul,
						float **d_tss, float **d_tes, float **d_tep, float **d_p, float **d_q, float **d_r, int verbose);
void cuda_freeCoefFields(modPar mod, float *d_rox, float *d_roz, float *d_l2m, float *d_lam, float *d_mul,
						float *d_tss, float *d_tes, float *d_tep, float *d_p, float *d_q, float *d_r, int verbose);
void cuda_initMainFields(modPar mod, float **d_vx, float **d_vz, float **d_tzz, float **d_txz, float **d_txx, int verbose);
void cuda_freeMainFields(modPar mod, float *d_vx, float *d_vz, float *d_tzz, float *d_txz, float *d_txx, int verbose);
void cuda_initRec(modPar mod, recPar rec, float **d_rec_vx, float **d_rec_vz, float **d_rec_p, float **d_rec_txx, 
				  float **d_rec_tzz, float **d_rec_txz, float **d_rec_pp, float **d_rec_ss, 
				  float **d_rec_udp, float **d_rec_udvz, int verbose);
void cuda_freeRec(recPar rec, float *d_rec_vx, float *d_rec_vz, float *d_rec_p, float *d_rec_txx, 
				  float *d_rec_tzz, float *d_rec_txz, float *d_rec_pp, float *d_rec_ss, 
				  float *d_rec_udp, float *d_rec_udvz, int verbose);
void cuda_init_d_src_nwav(wavPar wav, float **src_nwav, float **d_src_nwav, int verbose);
void cuda_free_d_src_nwav(float *d_src_nwav, int verbose);


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

// aux print functions
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

// Tests
__global__ void kernel_printf_srcPar2(srcPar *d_src);

#endif

