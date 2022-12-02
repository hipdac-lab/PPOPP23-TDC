#include <cudnn.h>
#include <stdio.h>
#include <cuda.h>
#include <malloc.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <sys/types.h>
#include <errno.h>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#define TH 1
#define TW 2
#define TC 16
#define C 192
#define N 96
#define H 14
#define W 14

#define TCS ((C-1)/TC + 1)
#define THS ((H-1)/TH + 1)
#define TWS ((W-1)/TW+1)
#define WPAD (TWS*TW + 2)
#define R 3
#define S 3


using namespace std;
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<endl;
        exit(-1);
    }
}
extern "C" __global__ void default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[6144];
  __shared__ float kernel_shared[5184];
  float pad_temp_shared_local[2];
  float kernel_shared_local[4];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + (((((int)threadIdx.x) * 74) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 1) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 2) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 3) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 4) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 5) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 6) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 7) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 8) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 9) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 10) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 10) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 11) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 11) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 12) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 12) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 13) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 13) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 14) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 14) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 15) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 15) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 16) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 16) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 17) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 17) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 18) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 18) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 19) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 19) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 20) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 20) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 21))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 21) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 21) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 22))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 22) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 22) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 23))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 23) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 23) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 24))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 24) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 24) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 25))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 25) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 25) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 26))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 26) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 26) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 27))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 27) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 27) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 28))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 28) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 28) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 29))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 29) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 29) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 30))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 30) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 30) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 31))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 31) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 31) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 32))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 32) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 32) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 32) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 32) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 33))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 33) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 33) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 33) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 33) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 34))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 34) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 34) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 34) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 34) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 35))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 35) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 35) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 35) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 35) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 36))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 36) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 36) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 36) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 36) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 37))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 37) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 37) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 37) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 37) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 38))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 38) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 38) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 38) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 38) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 39))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 39) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 39) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 39) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 39) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 40))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 40) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 40) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 40) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 40) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 41))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 41) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 41) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 41) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 41) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 42))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 42) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 42) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 42) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 42) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 43))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 43) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 43) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 43) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 43) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 44))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 44) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 44) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 44) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 44) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 45))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 45) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 45) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 45) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 45) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 46))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 46) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 46) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 46) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 46) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 47))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 47) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 47) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 47) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 47) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 48))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 48) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 48) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 48) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 48) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 49))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 49) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 49) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 49) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 49) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 50))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 50) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 50) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 50) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 50) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 51))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 51) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 51) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 51) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 51) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 52))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 52) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 52) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 52) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 52) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 53))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 53) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 53) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 53) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 53) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 54))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 54) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 54) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 54) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 54) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 55))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 55) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 55) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 55) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 55) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 56))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 56) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 56) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 56) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 56) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 57))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 57) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 57) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 57) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 57) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 58))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 58) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 58) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 58) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 58) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 59))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 59) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 59) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 59) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 59) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 60))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 60) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 60) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 60) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 60) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 61))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 61) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 61) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 61) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 61) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 62) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 62) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6082) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1986) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 62))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 62) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 62) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 62) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 62) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 63) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 63) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6081) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1985) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 63))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 63) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 63) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 63) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 63) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) * 74) >> 6)) < 95) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + ((((int)threadIdx.x) * 74) >> 4)) < 380) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6080) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1984) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 64))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + (((((int)threadIdx.x) * 74) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 181))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 65) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 65) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6079) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1983) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 65))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 65) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 66) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 66) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6078) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1982) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 66))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 66) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 67) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 67) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6077) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1981) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 67))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 67) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 68) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 68) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6076) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1980) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 68))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 68) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 69) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 69) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6075) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1979) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 69))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 69) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 70) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 70) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6074) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1978) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 70))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 70) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 71) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 71) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6073) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1977) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 71))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 71) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 72) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 72) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6072) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1976) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 72))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 72) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 16)) + (((((int)threadIdx.x) * 74) + 73) >> 6)) < 96) {
      if ((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 64)) + (((((int)threadIdx.x) * 74) + 73) >> 4)) < 384) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) < 6071) {
          if (((((int)threadIdx.y) * 1024) + (((int)threadIdx.x) * 74)) < 1975) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 74)) + 73))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((((rc_outer * 18816) + (((int)threadIdx.z) * 6272)) + (((int)threadIdx.y) * 3136)) + ((((((int)threadIdx.x) * 74) + 73) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    kernel_shared[((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)))] = kernel[((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 1))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 1))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 2))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 2))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 3))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 3))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 4))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 4))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 5))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 5))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 6))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 6))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 7))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 7))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 8))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 8))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 9))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 9))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 10))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 10))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 11))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 11))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 12))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 12))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 13))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 13))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 14))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 14))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 15))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 15))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 16))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 16))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 17))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 17))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 18))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 18))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 19))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 19))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 20))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 20))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 21))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 21))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 22))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 22))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 23))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 23))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 24))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 24))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 25))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 25))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 26))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 26))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 27))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 27))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 28))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 28))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 29))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 29))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 30))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 30))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 31))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 31))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 32))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 32))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 33))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 33))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 34))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 34))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 35))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 35))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 36))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 36))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 37))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 37))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 38))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 38))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 39))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 39))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 40))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 40))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 41))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 41))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 42))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 42))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 43))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 43))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 44))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 44))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 45))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 45))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 46))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 46))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 47))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 47))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 48))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 48))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 49))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 49))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 50))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 50))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 51))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 51))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 52))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 52))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 53))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 53))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 54))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 54))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 55))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 55))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 56))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 56))];
    kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 57))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 57))];
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 62) + 58) / 864)) + ((int)threadIdx.y)) < 6) {
      if ((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 96)) + (((((int)threadIdx.x) * 62) + 58) / 9)) < 576) {
        if ((((((int)threadIdx.z) * 576) + (((int)threadIdx.y) * 288)) + (((((int)threadIdx.x) * 62) + 58) / 3)) < 1728) {
          if ((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) < 5126) {
            if (((((int)threadIdx.y) * 864) + (((int)threadIdx.x) * 62)) < 1670) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 58))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 58))];
              }
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 62) + 59) / 864)) + ((int)threadIdx.y)) < 6) {
      if ((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 96)) + (((((int)threadIdx.x) * 62) + 59) / 9)) < 576) {
        if ((((((int)threadIdx.z) * 576) + (((int)threadIdx.y) * 288)) + (((((int)threadIdx.x) * 62) + 59) / 3)) < 1728) {
          if ((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) < 5125) {
            if (((((int)threadIdx.y) * 864) + (((int)threadIdx.x) * 62)) < 1669) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 59))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 59))];
              }
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 62) + 60) / 864)) + ((int)threadIdx.y)) < 6) {
      if ((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 96)) + (((((int)threadIdx.x) * 62) + 60) / 9)) < 576) {
        if ((((((int)threadIdx.z) * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) * 62) / 3)) < 1708) {
          if ((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) < 5124) {
            if (((((int)threadIdx.y) * 864) + (((int)threadIdx.x) * 62)) < 1668) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 60))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 60))];
              }
            }
          }
        }
      }
    }
    if ((((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 62) + 61) / 864)) + ((int)threadIdx.y)) < 6) {
      if ((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 96)) + (((((int)threadIdx.x) * 62) + 61) / 9)) < 576) {
        if ((((((int)threadIdx.z) * 576) + (((int)threadIdx.y) * 288)) + (((((int)threadIdx.x) * 62) + 61) / 3)) < 1728) {
          if ((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) < 5123) {
            if (((((int)threadIdx.y) * 864) + (((int)threadIdx.x) * 62)) < 1667) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1728) + (((int)threadIdx.y) * 864)) + (((int)threadIdx.x) * 62)) + 61))] = kernel[(((((((((int)blockIdx.z) * 10368) + (((int)threadIdx.z) * 3456)) + (((int)threadIdx.y) * 1728)) + (rc_outer * 864)) + (((int)threadIdx.x) * 62)) + 61))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 48; ++rc_inner_outer) {
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 64))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 9))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 864))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 873))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 1))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 65))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 1))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 10))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 865))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 874))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 66))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 2))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 11))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 866))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 875))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 16))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 80))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 3))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 12))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 867))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 876))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 17))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 81))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 4))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 13))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 868))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 877))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 18))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 82))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 5))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 14))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 869))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 878))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 32))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 96))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 6))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 15))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 870))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 879))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 33))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 97))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 7))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 16))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 871))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 880))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 34))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x)) + 98))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 8))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 17))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 872))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 1728) + (rc_inner_outer * 18)) + 881))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
    }
  }
  compute[((((((((int)blockIdx.z) * 1176) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 1176) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 196))] = compute_local[(1)];
}





class ConvGemm{
public:
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *kernel;
    void initialize();
    float *forward(float *input);
};
void ConvGemm::initialize(){

    cudaMalloc(&kernel,sizeof(float)*C*N*9);
    cudaMalloc(&this->output,sizeof(float)*N*H*W);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnCreateFilterDescriptor(&convKernelDescriptor);
    cudnnSetFilter4dDescriptor(convKernelDescriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/N,
            /*in_channels=*/C,
            /*kernel_height=*/R,
            /*kernel_width=*/S);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc,
            /*pad_height=*/1,
            /*pad_width=*/1,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          convInputDescriptor,
                                          convKernelDescriptor,
                                          &batch_size,
                                          &channels,
                                          &height,
                                          &width);
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/N,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            &workspace_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    unsigned int kernelSize = R*S*C*N;//kernel
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    for(int i=0;i<kernelSize;++i){
        this->cpuKernel[i] = 1.0f;
    }
    cudaMemcpy(kernel,cpuKernel,R*S*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(cpuKernel);
}
float * ConvGemm::forward(float *input) {
    cudaMemset(output, 0, 1*N*H*W*sizeof(float));
    checkCUDNN(cudnnConvolutionForward(convCudnn,
                                       &alpha,
                                       convInputDescriptor,
                                       input,
                                       convKernelDescriptor,
                                       kernel,
                                       convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       convOutputDescriptor,
                                       output));
    return output;
}

class ConvWinogradeNon{
public:
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *kernel;
    void initialize();
    float *forward(float *input);
};
void ConvWinogradeNon::initialize(){
    cudaMalloc(&kernel,sizeof(float)*C*N*9);
    cudaMalloc(&this->output,sizeof(float)*N*H*W);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnCreateFilterDescriptor(&convKernelDescriptor);
    cudnnSetFilter4dDescriptor(convKernelDescriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/N,
            /*in_channels=*/C,
            /*kernel_height=*/R,
            /*kernel_width=*/S);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc,
            /*pad_height=*/1,
            /*pad_width=*/1,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          convInputDescriptor,
                                          convKernelDescriptor,
                                          &batch_size,
                                          &channels,
                                          &height,
                                          &width);
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/N,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                            &workspace_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    unsigned int kernelSize = R*S*C*N;//kernel
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    for(int i=0;i<kernelSize;++i){
        this->cpuKernel[i] = 1.0f;
    }
    cudaMemcpy(kernel,cpuKernel,R*S*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(cpuKernel);
}
float * ConvWinogradeNon::forward(float *input) {
    cudaMemset(output, 0, 1*N*H*W*sizeof(float));
    checkCUDNN(cudnnConvolutionForward(convCudnn,
                                       &alpha,
                                       convInputDescriptor,
                                       input,
                                       convKernelDescriptor,
                                       kernel,
                                       convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       convOutputDescriptor,
                                       output));
    return output;
}
class ConvFFT{
public:
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *kernel;
    void initialize();
    float *forward(float *input);
};
void ConvFFT::initialize(){

    cudaMalloc(&kernel,sizeof(float)*C*N*9);
    cudaMalloc(&this->output,sizeof(float)*N*H*W);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnCreateFilterDescriptor(&convKernelDescriptor);
    cudnnSetFilter4dDescriptor(convKernelDescriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/N,
            /*in_channels=*/C,
            /*kernel_height=*/R,
            /*kernel_width=*/S);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc,
            /*pad_height=*/1,
            /*pad_width=*/1,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    int batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          convInputDescriptor,
                                          convKernelDescriptor,
                                          &batch_size,
                                          &channels,
                                          &height,
                                          &width);
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/N,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                                            &workspace_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    unsigned int kernelSize = R*S*C*N;//kernel
    this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
    for(int i=0;i<kernelSize;++i){
        this->cpuKernel[i] = 1.0f;
    }
    cudaMemcpy(kernel,cpuKernel,R*S*C*N*sizeof(float),cudaMemcpyHostToDevice);
    free(cpuKernel);
}
float * ConvFFT::forward(float *input) {
    cudaMemset(output, 0, 1*N*H*W*sizeof(float));
    checkCUDNN(cudnnConvolutionForward(convCudnn,
                                       &alpha,
                                       convInputDescriptor,
                                       input,
                                       convKernelDescriptor,
                                       kernel,
                                       convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       convOutputDescriptor,
                                       output));
    return output;
}
__device__ void load_input_2_shared_memory(float *input, float *shared_input, unsigned int h_start,
                                           unsigned int h_end, unsigned int h_offset, unsigned int c_start,
                                           unsigned int warp_id, unsigned int lane_id, unsigned int warp_size){
    switch(h_offset){
        case 0:
            for(unsigned int c = warp_id; c<TC; c+=TWS){
        for(unsigned int i=lane_id; i<(h_end - h_start) * W; i+=warp_size){
            unsigned int r = i/W;
            unsigned int s = i%W;
            shared_input[c*(TH + 2)*(WPAD) + r * WPAD + s + 1] = input[(c_start + c) * H * W + h_start * W + i];
        }
    }
            break;
        case 1:
            for(unsigned int c = warp_id; c<TC; c+=TWS){
        for(unsigned int i=lane_id; i<(h_end - h_start) * W; i+=warp_size){
            unsigned int r = i/W;
            unsigned int s = i%W;
            shared_input[c*(TH + 2)*(WPAD) + (1 + r) * WPAD + s + 1] = input[(c_start + c) * H * W + h_start * W + i];
        }
    }
            break;
    }
}
__device__ __forceinline__ void switch_write_back(unsigned int write_h, unsigned int write_w, unsigned int h_out_start, unsigned int w_out_start, unsigned int n, float * outputs, float * temp_result){
	switch(write_h){
		case 1: 
 		switch(write_w){
			case 1:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 1; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 2:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 2; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
		} 
		break;
	}
}
        __global__ void conv2d(float * __restrict__ input,const float * __restrict__ kernel, float * __restrict__ outputs){
    extern __shared__ float shared_input[];
    const unsigned int tile_id = blockIdx.x;
    const unsigned int tc_id = tile_id / THS;
    const unsigned int th_id = tile_id % THS;
    const unsigned int tw_id = threadIdx.x / N;
    const int h_out_start = th_id * TH;
    const int w_out_start = tw_id * TW;
    const unsigned int warp_id = tw_id;
    const unsigned int lane_id = threadIdx.x % N;
    float data_array[9];
    float temp_result[TH*TW] = {0.0f};
    for(unsigned int i=threadIdx.x;i<TC*(TH+2)*WPAD;i+=blockDim.x){
        shared_input[i] = 0.0f;
    }
    unsigned int n = lane_id;
    unsigned int c_offset = tc_id * TC;
    int h_offset = (h_out_start == 0)?1:0;
    int h_padded_start = h_out_start;
    int h_padded_end = min(h_padded_start + TH + 2, H + 2);
    int h_non_padded_start = max(h_out_start - 1, 0);
    int h_non_padded_end = min(H, h_padded_end - 1);
    __syncthreads();
    load_input_2_shared_memory(input, shared_input, h_non_padded_start, h_non_padded_end, h_offset, c_offset, warp_id, lane_id, N);
    __syncthreads();
#pragma unroll
    for(unsigned int c=0;c<TC;c++){
#pragma unroll
        for(unsigned int r=0;r<R;++r){
#pragma unroll
            for(unsigned int s=0;s<S;++s){
                data_array[r*S+s] = kernel[(c + c_offset)*N*9+r*3*N+s*N+n];
            }
        }
        		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];

    }
    switch_write_back(min(TH, H - h_out_start), min(TW, W - w_out_start), h_out_start, w_out_start, n, outputs, temp_result);
}

float check_diff(float *x, float *y, unsigned int size){
    float diff = 0.0f;
#pragma omp parallel for reduction(+ : diff)
    for(unsigned int i=0;i<size;++i){
        diff += abs(x[i] - y[i]);
    }
    return diff;
}
int main(void){
    float *input = new float[C*H*W];
    time_t t;
    float *matrix;
    cudaMalloc(&matrix,C*(TH+2)*(TW+2)*THS*TWS*sizeof(float));
    cudaMemset(matrix,0,C*(TH+2)*(TW+2)*THS*TWS*sizeof(float));
    srand((unsigned) time(&t));
    for(int i =0;i<C*H*W;++i){
        input[i] = rand() % 10;
    }
    float *device_input;
    cudaMalloc(&device_input,C*H*W*sizeof(float));
    cudaMemcpy(device_input,input,C*H*W*sizeof(float),cudaMemcpyHostToDevice);
    float *K = new float[C*N*9];
    for(int i=0;i<C*N*9;++i){
        K[i] = 1.0f;
    }

    ConvGemm convGemm;
    convGemm.initialize();
    ConvWinogradeNon convWinogradeNon;
    convWinogradeNon.initialize();
    ConvFFT convFFT;
    convFFT.initialize();

    float *out_cudnn;
    float *out_cudnn_host = new float[N*H*W];
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    out_cudnn = convGemm.forward(device_input);
    cudaMemcpy(out_cudnn_host,out_cudnn,N*H*W*sizeof(float),cudaMemcpyDeviceToHost);
    out_cudnn = convFFT.forward(device_input);
    out_cudnn = convWinogradeNon.forward(device_input);

    float *device_K;
    float *device_out;
    cudaMalloc(&device_out,H*W*N*sizeof(float));
    cudaMemset(device_out,0,H*W*N*sizeof(float));
    cudaMalloc(&device_K,C*N*9*sizeof(float));
    cudaMemcpy(device_K,K,C*N*9*sizeof(float),cudaMemcpyHostToDevice);

    cudaEventRecord(event_start);
    convGemm.forward(device_input);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float cudnnGemmTime;
    cudaEventElapsedTime(&cudnnGemmTime, event_start, event_stop);


    cudaEventRecord(event_start);
    convWinogradeNon.forward(device_input);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float cudnnWinogradeTimeNon;
    cudaEventElapsedTime(&cudnnWinogradeTimeNon, event_start, event_stop);

    cudaEventRecord(event_start);
    convFFT.forward(device_input);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float cudnnFFTTime;
    cudaEventElapsedTime(&cudnnFFTTime, event_start, event_stop);


        dim3 grid(1,7,16);

                dim3 block(14,2,3);

    cudaEventRecord(event_start);
    default_function_kernel0<<<grid, block>>>(device_input, device_K, device_out);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_tvm;
    cudaEventElapsedTime(&time_tvm, event_start, event_stop);
    float *out_tvm = new float[N*H*W];
    cudaMemcpy(out_tvm,device_out,N*H*W*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemset(device_out, 0, sizeof(float)*N*H*W);

    chkerr(cudaFuncSetAttribute(conv2d,cudaFuncAttributeMaxDynamicSharedMemorySize, TC*(TH+2)*(WPAD)*4));
    cudaEventRecord(event_start);
    conv2d<<<TCS*THS, N * TWS, TC*(TH+2)*(WPAD)*4>>>(device_input, device_K, device_out);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_tdc;
    cudaEventElapsedTime(&time_tdc, event_start, event_stop);
    float *out_tdc = new float[N*H*W];
    cudaMemcpy(out_tdc,device_out,N*H*W*sizeof(float),cudaMemcpyDeviceToHost);

    ofstream outfile;
    char buffer[1000];
    int ret = sprintf(buffer,"%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",N,C,H,W,
            cudnnFFTTime,cudnnWinogradeTimeNon,cudnnGemmTime,time_tvm,time_tdc,
            cudnnFFTTime/time_tdc,cudnnWinogradeTimeNon/time_tdc,cudnnGemmTime/time_tdc,time_tvm/time_tdc);
    outfile.open("../../evaluation_outcome/A100-layers-eval-modeling.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_tvm, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<
                                   cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<","<<difference<<endl;
    return 0;
}


