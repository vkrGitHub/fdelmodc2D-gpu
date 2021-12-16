#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> //for cuda_printWhichGPU
#include <string.h>

#include "par.h"
#ifdef _CRAYMPP
#include <intrinsics.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
SPECIAL PRINT

*/
void printfgpu(char *fmt, ...)
{
    va_list args;

    if (EOF == fflush(stderr)) {
        fprintf(stderr, "\nprintfgpu: fflush failed on stderr");
    }
    cudaDeviceProp prop;
    int idev; 
    cudaGetDevice(&idev);
    cudaGetDeviceProperties(&prop, idev);
    //fprintf(stderr, "\n%s (Device %d): ", prop.name, idev);
    fprintf(stdout, "\n%s (Device %d): ", prop.name, idev);
    //fprintf(stderr, "    %s: ", xargv[0]);
#ifdef _CRAYMPP
        fprintf(stderr, "PE %d: ", _my_pe());
#elif defined(SGI)
        fprintf(stderr, "PE %d: ", mp_my_threadnum());
#endif
    va_start(args,fmt);
    //vfprintf(stderr, fmt, args);
    vfprintf(stdout, fmt, args);
    va_end(args);
    //fprintf(stderr, "\n");
    fprintf(stdout, "\n");
    return;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
CUDA ERROR FUNCTIONS

*/

void wrap_cudaGetLastError (const char *msg) 
/* Wrapper for cudaGetLastError function
Useful for debugging cudacalls without "polluting" the code 
*/
{
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err) { 
	fprintf (stdout, "Cuda error: %s: %s (code %d) \n", msg, cudaGetErrorString (err), err); 
	exit(0);   
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
CUDA PERFORMANCE FUNCTIONS
Some performance metrics for cuda kernels.
Originally intended for single-gpu usage.

References:
https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
Cheng, J., Grossman, M. and McKercher, T., 2014. Professional Cuda C Programming. John Wiley & Sons.

*/

#include "fdelmodc.h"

//OVERALL DEVICE PROPERTIES FUNCTIONS
void wrap_cudaGetDeviceProperties(){
/*
Prints main device properties, for all devices in the node
*/

    cudaDeviceProp   prop;
    int count;

    cudaGetDeviceCount( &count );

    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i ) ;
        printf( "    --- General Information for device %d ---\n", i );
        printf( "Name:   %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:   %.2f MHz\n", (float) prop.clockRate/1000. ); // Compare with nvidia-smi -q -d clock
        printf( "Memory Clock rate:   %.2f MHz\n", (float) prop.memoryClockRate/1000. );
        printf( "Memory Bus width:   %d bits\n", prop.memoryBusWidth );
		printf( "Device copy overlap:   %s", prop.deviceOverlap? "Enabled\n":"Disabled\n" );
        printf( "Kernel execution timeout :   %s", prop.kernelExecTimeoutEnabled? "Enabled\n":"Disabled\n" );

        printf( "\n    --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:   %ld bytes\n", prop.totalGlobalMem );
        printf( "Total constant Mem:   %ld bytes\n", prop.totalConstMem );
        //printf( "Max mem pitch:   %ld\n", prop.memPitch );
        //printf( "Texture Alignment:   %ld\n", prop.textureAlignment );

        printf( "\n   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:   %d\n", prop.multiProcessorCount );
		printf( "Max threads per MP: %d\n", prop.maxThreadsPerMultiProcessor );
		printf( "Max 'physical' threads: %d\n", prop.multiProcessorCount*prop.maxThreadsPerMultiProcessor);
        printf( "Shared mem per MP:   %ld bytes\n", prop.sharedMemPerBlock );
        printf( "Registers per MP:   %d\n", prop.regsPerBlock );
        printf( "Threads in warp:   %d\n", prop.warpSize );
        printf( "Max threads per block:   %d\n", prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:   (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:   (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
        printf( "\n" );

	}
}


void wrap_cudaMemGetInfo(char *msg){
/*
Prints the output of cudaMemGetInfo. Tells used bytes, free bytes, total bytes (global mem)
*/
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status;
    int idev;

    cudaGetDevice(&idev); // device is set from outside this function
    cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("(Device %d) Error: cudaMemGetInfo fails, %s \n", idev, cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printfgpu("Meminfo %s:\n\t\t\t\tUsed = %.2fMB\n\t\t\t\tFree = %.2fMB\n\t\t\t\tTotal = %.2fMB", msg, used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    // printf("(Device %d) GPU memory usage: used = %.2fKB, free = %.2fKB, total = %.2fKB\n", idev, used_db/1024.0, free_db/1024.0, total_db/1024.0);
}


// // CUDA write/read/append functions
// void cuda_printWhichGPU2(char *text){
// /*
// Get current GPU number and name, print it after "text"
// */
//     cudaDeviceProp prop;
//     int idev; 
//     cudaGetDevice(&idev);
//     cudaGetDeviceProperties(&prop, idev);
//     fprintf(stdout, "\n%s (Device %d): %s", prop.name, idev, text);    
// }


// // CUDA write/read/append functions
// void cuda_printWhichGPU(char *fmt, ...){
// /*
// Get current GPU number and name, print it after "text"
// */
//     va_list args;

//     if (EOF == fflush(stdout)) {
//         fprintf(stderr, "\ncuda_printWhichGPU: fflush failed on stderr");
//     }

//     fprintf(stderr, "    %s: ", xargv[0]);

//     cudaDeviceProp prop;
//     int idev; 
//     cudaGetDevice(&idev);
//     cudaGetDeviceProperties(&prop, idev);
//     fprintf(stdout, "\n%s (Device %d): ", prop.name, idev);
    
//     va_start(args,fmt);
//     vfprintf(stdout, fmt, args);
//     va_end(args);
//     fprintf(stderr, "\n");
//     return;
// }


void cuda_wrap_fread(void *d_input, int size, int stride, char *filename){
    void *input;
    input = malloc(size*stride);

    FILE *fp;
    fp = fopen(filename, "r");
    fread(input, size, stride, fp);
    fclose(fp); 

    cudaMemcpy(d_input, input, size*stride, cudaMemcpyHostToDevice);    

    free(input);

    cudaDeviceSynchronize();
}

void cuda_wrap_fwrite(void *d_input, int size, int stride, char *filename){
/*
Wrapper to cudamemcpyD2H and save to a new file
*/
    void *input;
    input = malloc(size*stride);
    cudaMemcpy(input, d_input, size*stride, cudaMemcpyDeviceToHost);    

    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(input, size, stride, fp);
    fclose(fp); 

    free(input);
}

void cuda_wrap_fwrite_append(void *d_input, int size, int stride, char *filename){
/*
Wrapper to cudamemcpyD2H and append to a file
*/
    void *input;
    input = malloc(size*stride);
    cudaMemcpy(input, d_input, size*stride, cudaMemcpyDeviceToHost);    

    FILE *fp;
    fp = fopen(filename, "a");
    fwrite(input, size, stride, fp);
    fclose(fp); 

    free(input);
}



void wrap_fread(void *input, int size, int stride, char *filename){
/*
Wrapper to load file to array
The calling order with 'size' before 'stride' follows fread's convention
*/
    FILE *fp;
    fp = fopen(filename, "r");
    fread(input, size, stride, fp);
    fclose(fp); 
} 

void wrap_fwrite(void *input, int size, int stride, char *filename){
/*
Wrapper to write array to file
The calling order with 'size' before 'stride' follows fwrite's convention
*/

    FILE *fp;
    fp = fopen(filename, "w");
    fwrite(input, size, stride, fp);
    fclose(fp); 
}

void wrap_fwrite_append(void *input, int size, int stride, char *filename){
    FILE *fp;
    fp = fopen(filename, "a");
    fwrite(input, size, stride, fp);
    fclose(fp); 
}


void save_dif_hd_1dF(int N, float *h_v, float *d_v, char *name){
/*
Prints stuff about differnce of host and device variable
Also saves into a file
*/
    // Declare
    float *h_tmp, *h_dif;
    float acm;
    int id;
    FILE *fp;
    
    // Alloc
    h_tmp = (float*)malloc(N*sizeof(float));
    h_dif = (float*)calloc(N, sizeof(float));
    
    // Init
    cudaMemcpy(h_tmp, d_v, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Process
    fp = fopen(name, "w");
    acm = 0.0;
    for(id=0; id<N; id++){
        acm += fabs(h_v[id]-h_tmp[id]);
        h_dif[id] = h_v[id]-h_tmp[id];
    }   
    fwrite(h_dif, N, sizeof(float), fp);
    fclose(fp);

    // Output2user
    printf("save_dif_hd_1dF %s SumErr:%f AvgErr:%f\n", name, acm, acm/N);

    // Free
    free(h_tmp);
    free(h_dif);
}

void app_dif_hd_1dF(int N, float *h_v, float *d_v, char *name){
/*
Prints stuff about differnce of host and device variable
Also appends into a file named 'name'
*/
    // Declare
    float *h_tmp, *h_dif;
    float acm;
    int id;
    FILE *fp;
    
    // Alloc
    h_tmp = (float*)malloc(N*sizeof(float));
    h_dif = (float*)calloc(N, sizeof(float));
    
    // Init
    cudaMemcpy(h_tmp, d_v, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Process
    fp = fopen(name, "a");
    acm = 0.0;
    for(id=0; id<N; id++){
        acm += fabs(h_v[id]-h_tmp[id]);
        h_dif[id] = h_v[id]-h_tmp[id];
    }   
    fwrite(h_dif, N, sizeof(float), fp);
    fclose(fp);

    // Output2user
    printf("app_dif_hd_1dF %s SumErr:%f AvgErr:%f\n", name, acm, acm/N);

    // Free
    free(h_tmp);
    free(h_dif);
}

void save_dif_hd_2dF(int nz, int nx, float *h_v, float *d_v, char *name){
/*
Prints stuff about differnce of host and device variable
Also saves into a file
*/
    // Declare
    float *h_tmp, *h_dif;
    float acm;
    int id, iz, ix;
    int N = nz*nx;
    FILE *fp;
    
    // Alloc
    h_tmp = (float*)malloc(N*sizeof(float));
    h_dif = (float*)calloc(N, sizeof(float));
    
    // Init
    cudaMemcpy(h_tmp, d_v, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Process
    fp = fopen(name, "w");
    acm = 0.0;
    for(ix=0; ix<nx; ix++){
        for(iz=0; iz<nz; iz++){
            id = ix*nz + iz;
            
            acm += fabs(h_v[id]-h_tmp[id]);
            h_dif[id] = h_v[id]-h_tmp[id];
        }
    }   
    fwrite(h_dif, N, sizeof(float), fp);
    fclose(fp);

    // Output2user
    printf("save_dif_hd_2dF %s SumErr:%f AvgErr:%f\n", name, acm, acm/N);

    // Free
    free(h_tmp);
    free(h_dif);
}

void app_dif_hd_2dF(int nz, int nx, float *h_v, float *d_v, char *name){
/*
Prints stuff about differnce of host and device variable
Also saves into a file
*/
    // Declare
    float *h_tmp, *h_dif;
    float acm;
    int id, iz, ix;
    int N = nz*nx;
    FILE *fp;
    
    // Alloc
    h_tmp = (float*)malloc(N*sizeof(float));
    h_dif = (float*)calloc(N, sizeof(float));
    
    // Init
    cudaDeviceSynchronize();
    cudaMemcpy(h_tmp, d_v, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Process
    fp = fopen(name, "a");
    acm = 0.0;
    for(ix=0; ix<nx; ix++){
        for(iz=0; iz<nz; iz++){
            id = ix*nz + iz;
            
            acm += fabs(h_v[id]-h_tmp[id]);
            h_dif[id] = h_v[id]-h_tmp[id];
        }
    }   
    cudaDeviceSynchronize();
    fwrite(h_dif, N, sizeof(float), fp);
    fclose(fp);

    // Output2user
    printf("app_dif_hd_2dF %s SumErr:%f AvgErr:%f\n", name, acm, acm/N);

    // Free
    free(h_tmp);
    free(h_dif);
}


