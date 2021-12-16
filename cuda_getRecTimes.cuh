#ifndef _CUDA_GETRECTIMES_CUH
#define _CUDA_GETRECTIMES_CUH

void cuda_init_getRecTimes(modPar mod, bndPar bnd);
void cuda_destroy_getRecTimes();
void cuda_getRecTimes(modPar mod, recPar rec, bndPar bnd, 
				modPar *d_mod, recPar *d_rec, bndPar *d_bnd, 
				int itime, int isam, float *d_vx, float *d_vz, 
				float *d_tzz, float *d_txx, float *d_txz, float *d_l2m, 
				float *d_rox, float *d_roz, float *d_rec_vx, float *d_rec_vz, 
				float *d_rec_txx, float *d_rec_tzz, float *d_rec_txz, float *d_rec_p, 
				float *d_rec_pp, float *d_rec_ss, float *d_rec_udp, float *d_rec_udvz, int verbose);
void cuda_print_getRecTimes_time();

#endif