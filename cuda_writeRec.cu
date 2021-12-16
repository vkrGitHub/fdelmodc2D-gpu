#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include "segy.h"
#include "fdelmodc.h"

#include "cuda_myutils.cuh"
#include "cuda_fileOpen.cuh" //Thorbecke's utils + cuda_traceWrite

/**
*  Writes the receiver array(s) to output file(s)
*
*   AUTHOR:
*           Jan Thorbecke (janth@xs4all.nl)
*           The Netherlands 
**/

static int ikey=1; //del

// CUDA timers
static cudaEvent_t t0, t1;
static float tmsec = 0; 
static float tall = 0; 
static int nrun = 0;

// CUDA Streams for applying all interpolations simultaneously, if possible
static int nstream = 8; //for types p, txx, tzz, txz, pp, vz, vx, ud
static cudaStream_t *streams;

void cuda_init_writeRecTimes(){
/*
Init vars for cuda_writeRecTimes
*/
    // Create timers
    cudaEventCreate(&t0); 
    cudaEventCreate(&t1); 

    // Create streams
    streams = (cudaStream_t*)malloc(nstream*sizeof(cudaStream_t));
    for(int i=0; i<nstream; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    wrap_cudaGetLastError("init_cuda_writeRecTimes");
    // Tell user we are done
    printfgpu("init_cuda_getRecTimes.");
}//end cuda_init_getRecTimes

void cuda_destroy_writeRecTimes(){
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
    printfgpu("destroy_cuda_writeRecTimes.");
}//end cuda_destroy_getRecTimes


#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

int cuda_writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav, int ixsrc, int izsrc, int nsam, int ishot, int fileno, 
             float *d_rec_vx, float *d_rec_vz, float *d_rec_txx, float *d_rec_tzz, float *d_rec_txz, 
             float *d_rec_p, float *d_rec_pp, float *d_rec_ss, float *d_rec_udp, float *d_rec_udvz, int verbose)
{
    FILE    *fpvx, *fpvz, *fptxx, *fptzz, *fptxz, *fpp, *fppp, *fpss, *fpup, *fpdown;
    float *rec_up, *rec_down, *trace, *rec_vze, *rec_pe;
    float dx, dt, cp, rho, fmin, fmax;
    int irec, ntfft, nfreq, nkx, xorig, ix, iz, it, ibndx;
    int append, vznorm, sx;
    double ddt;
    char number[16], filename[1024];
    segy hdr;
    
    if (!rec.n) return 0;
    if (ishot) append=1;
    else append=0;

    // Time 
    cudaEventRecord(t0); 

    /* if the total number of samples exceeds rec_ntsam then a new (numbered) file is opened */
    /* fileno has a non-zero value (from fdelmodc.c) if the number of samples exceeds rec_ntsam. */
    char gpufname[400];
    strcpy(gpufname, rec.file_rcv);
    name_ext(gpufname, "-gpu");
    // sprintf(gpufname, "gpu-%s", rec.file_rcv);
    //strcpy(filename, rec.file_rcv);
    strcpy(filename, gpufname);
    if (fileno) {
        sprintf(number,"_%03d",fileno);
        name_ext(filename, number);
    }
#ifdef MPI
    sx = (int)mod.x0+ixsrc*mod.dx;
    sprintf(number,"_%06d",sx);
    name_ext(filename, number);
#endif

    if (verbose>1) printf("cuda_writeRec Writing receiver data to file %s\n", filename);
    if (nsam != rec.nt && verbose) printf("cuda_writeRec Number of samples written to last file = %d\n",nsam);

    memset(&hdr,0,TRCBYTES);
    ddt = (double)mod.dt;/* to avoid rounding in 32 bit precision */
    dt  = (float)ddt*rec.skipdt;
    dx  = (rec.x[1]-rec.x[0])*mod.dx;
    hdr.dt     = (unsigned short)lround((((double)1.0e6*ddt*rec.skipdt)));
    hdr.scalco = -1000;
    hdr.scalel = -1000;
    hdr.sx     = 1000*(mod.x0+ixsrc*mod.dx);
    hdr.sdepth = 1000*(mod.z0+izsrc*mod.dz);
    hdr.selev  = (int)(-1000.0*(mod.z0+izsrc*mod.dz));
    hdr.fldr   = ishot+1;
    hdr.trid   = 1;
    hdr.ns     = nsam;
    hdr.trwf   = rec.n;
    hdr.ntr    = rec.n;
    if (mod.grid_dir) { /* reverse time modeling */
        hdr.f1 = (-mod.nt+1)*mod.dt;
    }
    else {
        hdr.f1 = 0.0;
    }
    hdr.d1     = mod.dt*rec.skipdt;
    hdr.d2     = (rec.x[1]-rec.x[0])*mod.dx;
    hdr.f2     = mod.x0+rec.x[0]*mod.dx;

    if (rec.type.vx)  fpvx  = fileOpen(filename, "_rvx", append);
    if (rec.type.vz)  fpvz  = fileOpen(filename, "_rvz", append);
    if (rec.type.p)   fpp   = fileOpen(filename, "_rp", append);
    if (rec.type.txx) fptxx = fileOpen(filename, "_rtxx", append);
    if (rec.type.tzz) fptzz = fileOpen(filename, "_rtzz", append);
    if (rec.type.txz) fptxz = fileOpen(filename, "_rtxz", append);
    if (rec.type.pp)  fppp  = fileOpen(filename, "_rpp", append);
    if (rec.type.ss)  fpss  = fileOpen(filename, "_rss", append);

    /* decomposed wavefield */
    if (rec.type.ud && (mod.ischeme==1 || mod.ischeme==2) )  {
        printf("cuda_writeRec GPU UD decomposition recording for ischemes 1,2 not available!\n");
    }
    if (rec.type.ud && (mod.ischeme==3 || mod.ischeme==4) )  {
        printf("cuda_writeRec GPU UD decomposition recording for ischemes 3,4 not available!\n");
    }

    for (irec=0; irec<rec.n; irec++) {
        hdr.tracf  = irec+1;
        hdr.tracl  = ishot*rec.n+irec+1;
        hdr.gx     = 1000*(mod.x0+rec.x[irec]*mod.dx);
        hdr.offset = (rec.x[irec]-ixsrc)*mod.dx;
        hdr.gelev  = (int)(-1000*(mod.z0+rec.z[irec]*mod.dz));

        if (rec.type.vx) {
             cuda_traceWrite( &hdr, &d_rec_vx[irec*rec.nt], nsam, streams[0], fpvx) ;
        }
        if (rec.type.vz) {
             cuda_traceWrite( &hdr, &d_rec_vz[irec*rec.nt], nsam, streams[1], fpvz) ;
        }
        if (rec.type.p) {
             cuda_traceWrite( &hdr, &d_rec_p[irec*rec.nt], nsam, streams[2], fpp) ;
        }
        if (rec.type.txx) {
             cuda_traceWrite( &hdr, &d_rec_txx[irec*rec.nt], nsam, streams[3], fptxx) ;
        }
        if (rec.type.tzz) {
             cuda_traceWrite( &hdr, &d_rec_tzz[irec*rec.nt], nsam, streams[4], fptzz) ;
        }
        if (rec.type.txz) {
             cuda_traceWrite( &hdr, &d_rec_txz[irec*rec.nt], nsam, streams[5], fptxz) ;
        }
        if (rec.type.pp) {
             cuda_traceWrite( &hdr, &d_rec_pp[irec*rec.nt], nsam, streams[6], fppp) ;
        }
        if (rec.type.ss) {
             cuda_traceWrite( &hdr, &d_rec_ss[irec*rec.nt], nsam, streams[7], fpss) ;
        }
        if (rec.type.ud && mod.ischeme==1)  {
             printf("cuda_writeRec GPU UD decomposition recording not available!\n");
        }
    }

    if (rec.type.vx) fclose(fpvx);
    if (rec.type.vz) fclose(fpvz);
    if (rec.type.p) fclose(fpp);
    if (rec.type.txx) fclose(fptxx);
    if (rec.type.tzz) fclose(fptzz);
    if (rec.type.txz) fclose(fptxz);
    if (rec.type.pp) fclose(fppp);
    if (rec.type.ss) fclose(fpss);
    if (rec.type.ud) {
        // fclose(fpup); // GPU UD decomp not yet available
        // fclose(fpdown);
        free(rec_up);
        free(rec_down);
    }

    wrap_cudaGetLastError("after cuda_getRecTimes");

    cudaDeviceSynchronize();
    cudaEventRecord(t1); 
    cudaEventSynchronize(t1); 
    cudaEventElapsedTime(&tmsec, t0, t1);
    tall += tmsec*1E-3; \
    nrun ++;

    return 0;
}

void cuda_print_writeRecTimes_time(){
    printf("cuda_writeRecTimes ran %d times and took %.4f s total.\n", nrun, tall);
}


