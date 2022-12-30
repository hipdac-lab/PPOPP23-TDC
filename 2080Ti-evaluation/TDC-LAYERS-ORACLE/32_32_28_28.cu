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
#define TW 3
#define TC 16
#define C 32
#define N 32
#define H 28
#define W 28

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
  float compute_local[16];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[2304];
  float pad_temp_shared_local[12];
  float kernel_shared_local[48];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)))] = (((((1 <= ((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 5) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 6) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 7) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 8) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 9) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 10) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 11) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 12) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 13) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 14) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 15) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 16) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 16) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 16) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 16) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) & 15)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) >> 8)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) >> 4)) < 256) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) < 4079) {
          if (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) < 2031) {
            pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 17) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 1) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) >> 8)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) >> 4)) < 256) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) < 4078) {
          if (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) < 2030) {
            pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 18) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 2) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) >> 8)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) >> 4)) < 256) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) < 4077) {
          if (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) < 2029) {
            pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 19) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 3) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) >> 8)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) >> 4)) < 256) {
        if ((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) < 4076) {
          if (((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) < 2028) {
            if (((int)threadIdx.x) < 13) {
              pad_temp_shared[(((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 293)) + (((int)threadIdx.x) * 21)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) & 255) >> 4))) && (((((int)blockIdx.y) * 14) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) & 255) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)) < 29)) ? data[(((((((((rc_outer * 12544) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) >> 8) * 784)) + (((int)blockIdx.y) * 392)) + ((((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 20) & 255) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 293) + (((int)threadIdx.x) * 21)) + 4) & 15)) - 29))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    kernel_shared[((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 48) * 288)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) % 48) * 3)))];
    kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 1))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 48) * 288)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) % 48) * 3)) + 1))];
    kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 2))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 48) * 288)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) % 48) * 3)) + 2))];
    kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 3))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) % 48) * 3)))];
    kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 4))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) % 48) * 3)) + 1))];
    kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 5))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 1) % 48) * 3)) + 2))];
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 3)) < 256) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 766) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2298) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1146) {
              if ((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 6))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) % 48) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 3)) < 256) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 766) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2297) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1145) {
              if ((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 7))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) % 48) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 3)) < 256) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 766) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2296) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1144) {
              if ((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 8))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 2) % 48) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + (((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 3)) < 255) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 765) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2295) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1143) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 9))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) % 48) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + (((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 3)) < 255) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 765) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2294) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1142) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 10))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) % 48) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48)) < 16) {
      if (((((int)threadIdx.z) * 128) + (((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) / 3)) < 255) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 55)) + (((int)threadIdx.x) * 4)) < 765) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) < 2293) {
            if (((((int)threadIdx.y) * 165) + (((int)threadIdx.x) * 12)) < 1141) {
              if (((int)threadIdx.x) < 13) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 165)) + (((int)threadIdx.x) * 12)) + 11))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) / 48) * 288)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 55) + (((int)threadIdx.x) * 4)) + 3) % 48) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 1))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 17))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 18))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 256))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 257))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 258))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 272))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 273))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 274))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)))];
      kernel_shared_local[(24)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1152))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1))];
      kernel_shared_local[(25)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1153))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 2))];
      kernel_shared_local[(26)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1154))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 9))];
      kernel_shared_local[(27)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1161))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 10))];
      kernel_shared_local[(28)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1162))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 11))];
      kernel_shared_local[(29)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1163))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 144))];
      kernel_shared_local[(30)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1296))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 145))];
      kernel_shared_local[(31)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1297))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 146))];
      kernel_shared_local[(32)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1298))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 153))];
      kernel_shared_local[(33)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1305))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 154))];
      kernel_shared_local[(34)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1306))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 155))];
      kernel_shared_local[(35)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1307))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 288))];
      kernel_shared_local[(36)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1440))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 289))];
      kernel_shared_local[(37)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1441))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 290))];
      kernel_shared_local[(38)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1442))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 297))];
      kernel_shared_local[(39)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1449))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 298))];
      kernel_shared_local[(40)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1450))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 299))];
      kernel_shared_local[(41)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1451))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 432))];
      kernel_shared_local[(42)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1584))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 433))];
      kernel_shared_local[(43)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1585))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 434))];
      kernel_shared_local[(44)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1586))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 441))];
      kernel_shared_local[(45)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1593))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 442))];
      kernel_shared_local[(46)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1594))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 443))];
      kernel_shared_local[(47)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1595))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(24)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(24)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(30)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(30)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(36)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(36)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(42)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(42)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(25)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(25)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(31)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(31)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(37)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(37)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(43)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(43)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(26)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(26)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(32)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(32)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(38)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(38)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(44)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(44)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(27)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(27)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(9)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(33)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(33)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(39)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(39)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(21)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(45)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(45)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(28)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(28)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(10)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(34)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(34)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(40)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(40)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(22)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(46)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(46)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(29)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(29)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(11)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(35)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(35)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(41)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(41)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(23)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(47)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(47)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 17))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 18))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 32))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 33))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 34))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 272))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 273))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 274))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 288))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 289))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 290))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 3))];
      kernel_shared_local[(24)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1155))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 4))];
      kernel_shared_local[(25)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1156))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 5))];
      kernel_shared_local[(26)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1157))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 12))];
      kernel_shared_local[(27)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1164))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 13))];
      kernel_shared_local[(28)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1165))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 14))];
      kernel_shared_local[(29)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1166))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 147))];
      kernel_shared_local[(30)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1299))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 148))];
      kernel_shared_local[(31)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1300))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 149))];
      kernel_shared_local[(32)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1301))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 156))];
      kernel_shared_local[(33)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1308))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 157))];
      kernel_shared_local[(34)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1309))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 158))];
      kernel_shared_local[(35)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1310))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 291))];
      kernel_shared_local[(36)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1443))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 292))];
      kernel_shared_local[(37)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1444))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 293))];
      kernel_shared_local[(38)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1445))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 300))];
      kernel_shared_local[(39)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1452))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 301))];
      kernel_shared_local[(40)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1453))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 302))];
      kernel_shared_local[(41)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1454))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 435))];
      kernel_shared_local[(42)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1587))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 436))];
      kernel_shared_local[(43)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1588))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 437))];
      kernel_shared_local[(44)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1589))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 444))];
      kernel_shared_local[(45)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1596))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 445))];
      kernel_shared_local[(46)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1597))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 446))];
      kernel_shared_local[(47)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1598))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(24)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(24)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(30)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(30)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(36)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(36)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(42)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(42)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(25)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(25)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(31)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(31)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(37)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(37)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(43)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(43)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(26)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(26)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(32)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(32)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(38)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(38)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(44)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(44)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(27)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(27)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(9)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(33)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(33)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(39)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(39)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(21)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(45)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(45)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(28)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(28)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(10)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(34)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(34)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(40)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(40)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(22)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(46)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(46)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(29)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(29)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(11)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(35)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(35)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(41)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(41)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(23)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(47)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(47)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 32))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 33))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 34))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 48))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 49))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 50))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 288))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 289))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 290))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 304))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 305))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 512) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 306))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 6))];
      kernel_shared_local[(24)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1158))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 7))];
      kernel_shared_local[(25)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1159))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 8))];
      kernel_shared_local[(26)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1160))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 15))];
      kernel_shared_local[(27)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1167))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 16))];
      kernel_shared_local[(28)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1168))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 17))];
      kernel_shared_local[(29)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1169))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 150))];
      kernel_shared_local[(30)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1302))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 151))];
      kernel_shared_local[(31)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1303))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 152))];
      kernel_shared_local[(32)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1304))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 159))];
      kernel_shared_local[(33)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1311))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 160))];
      kernel_shared_local[(34)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1312))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 161))];
      kernel_shared_local[(35)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1313))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 294))];
      kernel_shared_local[(36)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1446))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 295))];
      kernel_shared_local[(37)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1447))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 296))];
      kernel_shared_local[(38)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1448))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 303))];
      kernel_shared_local[(39)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1455))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 304))];
      kernel_shared_local[(40)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1456))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 305))];
      kernel_shared_local[(41)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1457))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 438))];
      kernel_shared_local[(42)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1590))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 439))];
      kernel_shared_local[(43)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1591))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 440))];
      kernel_shared_local[(44)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1592))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 447))];
      kernel_shared_local[(45)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1599))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 448))];
      kernel_shared_local[(46)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1600))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 449))];
      kernel_shared_local[(47)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 18)) + 1601))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(24)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(24)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(30)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(30)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(36)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(36)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(42)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(42)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(25)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(25)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(31)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(31)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(37)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(37)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(43)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(43)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(26)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(26)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(32)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(32)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(38)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(38)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(44)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(44)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(27)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(27)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(9)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(33)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(33)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(39)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(39)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(21)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(45)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(45)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(28)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(28)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(10)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(34)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(34)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(40)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(40)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(22)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(46)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(46)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(29)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(29)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(11)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(35)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(35)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(41)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(41)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(23)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(47)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(47)]));
    }
  }
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 6272))] = compute_local[(8)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 28))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 6300))] = compute_local[(9)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 784))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 7056))] = compute_local[(10)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 812))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 7084))] = compute_local[(11)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 1568))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 7840))] = compute_local[(12)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 1596))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 7868))] = compute_local[(13)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 2352))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 8624))] = compute_local[(14)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 2380))] = compute_local[(7)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)) + 8652))] = compute_local[(15)];
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
                                            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
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
                                       CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
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
			case 3:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 3; ++tw) { 
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
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[0];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[8];

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


        dim3 grid(2,2,2);

        dim3 block(14,7,2);

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
    outfile.open("../../evaluation_outcome/2080Ti-layers-eval-oracle.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_cudnn_host, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<
                                   cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<","<<difference<<endl;
    return 0;
}


