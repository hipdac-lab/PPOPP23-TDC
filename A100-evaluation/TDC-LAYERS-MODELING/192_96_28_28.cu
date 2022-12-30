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
#define TW 4
#define TC 16
#define C 192
#define N 96
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
  float compute_local[8];
  __shared__ float pad_temp_shared[1152];
  __shared__ float kernel_shared[864];
  float pad_temp_shared_local[24];
  float kernel_shared_local[18];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 21) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 1) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 1) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 1) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 1) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 1) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 2) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 2) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 2) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 2) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 2) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 3) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 3) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 3) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 3) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 3) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 4) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 4) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 4) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 4) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 4) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 5) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 5) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 5) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 5) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 5) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 1) & 15))) && (((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 1) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 1) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 1) & 15) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 21) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 7) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 7) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 7) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 7) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 1) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 8) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 8) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 8) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 8) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 2) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 9) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 9) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 9) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 9) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 3) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 10) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 10) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 10) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 10) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 4) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 11) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 11) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 11) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 11) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 5) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 2) & 15))) && (((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 2) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 2) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 2) & 15) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 21) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 13) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 13) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 13) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 13) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 1) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 14) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 14) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 14) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 14) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 2) % 6)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 15) / 6)) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + (((((int)threadIdx.x) * 21) + 15) / 6)) < 192) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1137) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 273) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 15) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 15) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 3) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 15) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 15) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 3) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 16) / 6)) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + (((((int)threadIdx.x) * 21) + 16) / 6)) < 192) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1136) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 272) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 16) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 16) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 4) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 16) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 16) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 4) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 17) / 6)) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + (((((int)threadIdx.x) * 21) + 17) / 6)) < 192) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1135) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 271) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 17) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 17) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 5) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 17) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 17) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 5) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 3) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + ((((int)threadIdx.x) * 21) / 6)) < 189) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1134) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 270) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 3) & 15))) && (((((int)blockIdx.y) * 14) + ((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 3) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 21) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 3) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + (((((((int)threadIdx.y) * 7) + ((((int)threadIdx.x) * 21) / 6)) + 3) & 15) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 21) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 19) / 6)) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + (((((int)threadIdx.x) * 21) + 19) / 6)) < 192) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1133) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 269) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 19) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 19) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 19) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 19) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 1) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 20) / 6)) >> 4)) < 12) {
      if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) + (((((int)threadIdx.x) * 21) + 20) / 6)) < 192) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1132) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 268) {
            pad_temp_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 20) / 6)) & 15))) && (((((int)blockIdx.y) * 14) + (((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 20) / 6)) & 15)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 21) + 2) % 6)) < 29)) ? data[(((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 20) / 6)) >> 4) * 784)) + (((int)blockIdx.y) * 392)) + ((((((int)threadIdx.y) * 7) + (((((int)threadIdx.x) * 21) + 20) / 6)) & 15) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 21) + 2) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    kernel_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) / 108) * 1728)) + (rc_outer * 108)) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 1))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 1) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 1) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 2))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 2) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 2) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 3))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 3) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 3) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 4))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 4) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 4) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 5))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 5) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 5) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 6))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 6) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 6) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 7))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 7) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 7) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 8))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 8) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 8) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 9))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 9) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 9) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 10))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 10) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 10) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 11))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 11) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 11) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 12))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 12) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 12) % 108)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 13))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 13) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 13) % 108)))];
    if (((((int)threadIdx.z) * 2) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 14) / 108)) < 8) {
      if (((((int)threadIdx.z) * 24) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 14) / 9)) < 96) {
        if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 14) / 3)) < 288) {
          if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) < 850) {
            if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) < 202) {
              kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 14))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 14) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 14) % 108)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 15) / 108)) < 8) {
      if (((((int)threadIdx.z) * 24) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 15) / 9)) < 96) {
        if (((((int)threadIdx.z) * 72) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) / 3)) < 283) {
          if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) < 849) {
            if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) < 201) {
              if (((int)threadIdx.x) < 1) {
                kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 16)) + 15))] = kernel[((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 3456)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 15) / 108) * 1728)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 16)) + 15) % 108)))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 12) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 6))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 8))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 12))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 96))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 98))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 102))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 104))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 108))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 110))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 114))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 116))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 192))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 194))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 198))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 200))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 204))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 206))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 210))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 212))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 216))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 12))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 15))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 21))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 24))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 108))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 111))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 114))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 117))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 120))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 123))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 126))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 129))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 132))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 3))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 7))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 13))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 15))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 19))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 97))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 99))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 103))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 105))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 109))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 111))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 115))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 117))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 193))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 195))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 199))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 201))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 205))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 207))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 211))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 213))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 16))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 19))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 22))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 25))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 109))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 112))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 115))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 118))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 121))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 124))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 127))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 130))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 133))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 4))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 8))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 16))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 98))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 100))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 104))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 106))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 110))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 112))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 116))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 118))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 194))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 196))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 200))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 202))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 206))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 208))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 212))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 214))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 11))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 20))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 23))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 26))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 110))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 113))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 116))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 119))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 122))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 125))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 128))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 131))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 134))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 288))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 290))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 294))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 296))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 300))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 302))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 306))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 308))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 384))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 386))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 390))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 392))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 396))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 398))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 402))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 404))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 480))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 482))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 486))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 488))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 492))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 494))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 498))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 500))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 27))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 30))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 33))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 36))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 39))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 42))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 45))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 48))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 51))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 135))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 138))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 141))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 144))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 147))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 150))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 153))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 156))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 159))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 289))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 291))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 295))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 297))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 301))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 303))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 307))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 309))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 385))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 387))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 391))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 393))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 397))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 399))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 403))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 405))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 481))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 483))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 487))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 489))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 493))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 495))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 499))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 501))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 28))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 31))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 34))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 37))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 40))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 43))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 46))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 49))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 52))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 136))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 139))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 142))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 145))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 148))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 151))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 154))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 157))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 160))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 290))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 292))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 296))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 298))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 302))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 304))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 308))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 310))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 386))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 388))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 392))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 394))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 398))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 400))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 404))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 406))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 482))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 484))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 488))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 490))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 494))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 496))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 500))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 502))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 29))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 32))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 35))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 38))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 41))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 44))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 47))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 50))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 53))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 137))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 140))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 143))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 146))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 149))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 152))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 155))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 158))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 161))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 576))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 578))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 582))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 584))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 588))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 590))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 594))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 596))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 672))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 674))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 678))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 680))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 684))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 686))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 690))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 692))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 768))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 770))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 774))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 776))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 780))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 782))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 786))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 788))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 54))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 57))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 60))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 63))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 66))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 69))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 72))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 75))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 78))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 162))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 165))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 168))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 171))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 174))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 177))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 180))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 183))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 186))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 577))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 579))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 583))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 585))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 589))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 591))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 595))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 597))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 673))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 675))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 679))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 681))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 685))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 687))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 691))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 693))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 769))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 771))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 775))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 777))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 781))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 783))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 787))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 789))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 55))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 58))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 61))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 64))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 67))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 70))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 73))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 76))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 79))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 163))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 166))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 169))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 172))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 175))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 178))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 181))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 184))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 187))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 578))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 580))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 584))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 586))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 590))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 592))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 596))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 598))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 674))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 676))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 680))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 682))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 686))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 688))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 692))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 694))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 770))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 772))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 776))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 778))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 782))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 784))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 788))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 790))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 56))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 59))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 62))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 65))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 68))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 71))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 74))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 77))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 80))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 164))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 167))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 170))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 173))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 176))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 179))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 182))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 185))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 188))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 864))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 866))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 870))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 872))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 876))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 878))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 882))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 884))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 960))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 962))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 966))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 968))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 972))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 974))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 978))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 980))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1056))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1058))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1062))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1064))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1068))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1070))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1074))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1076))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 81))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 84))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 87))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 90))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 93))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 96))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 99))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 102))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 105))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 189))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 192))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 195))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 198))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 201))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 204))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 207))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 210))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 213))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 865))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 867))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 871))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 873))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 877))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 879))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 883))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 885))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 961))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 963))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 967))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 969))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 973))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 975))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 979))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 981))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1057))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1059))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1063))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1065))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1069))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1071))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1075))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1077))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 82))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 85))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 88))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 91))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 94))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 97))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 100))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 103))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 106))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 190))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 193))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 196))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 199))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 202))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 205))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 208))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 211))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 214))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 866))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 868))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 872))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 874))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 878))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 880))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 884))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 886))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 962))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 964))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 968))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 970))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 974))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 976))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 980))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 982))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1058))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1060))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1064))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1066))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1070))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1072))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1076))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + 1078))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 83))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 86))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 89))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 92))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 95))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 98))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 101))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 104))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 107))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 216) + 191))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 216) + 194))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 216) + 197))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 216) + 200))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 216) + 203))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 216) + 206))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 216) + 209))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 216) + 212))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 216) + 215))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(12)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(13)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(14)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
  }
  compute[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 2))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 28))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 30))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 784))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 786))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 812))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 1568)) + (((int)blockIdx.y) * 392)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 814))] = compute_local[(7)];
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
			case 4:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 4; ++tw) { 
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
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[0];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[8];

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


        dim3 grid(7,2,12);

                dim3 block(2,7,4);

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


