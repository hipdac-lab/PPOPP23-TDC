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
#define TH 3
#define TW 2
#define TC 16
#define C 96
#define N 64
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
  __shared__ float pad_temp_shared[504];
  __shared__ float kernel_shared[1152];
  float pad_temp_shared_local[6];
  float kernel_shared_local[24];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 9) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 9) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 9) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 1))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 1) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 1) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 1) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 1) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 1) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 1) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 2))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 2) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 2) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 2) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 2) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 2) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 2) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 3))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 3) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 3) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 3) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 3) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 3) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 3) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 3) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 4))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 4) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 4) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 4) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 4) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 4) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 4) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 4) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 5))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 5) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 5) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 5) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 5) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 5) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 5) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 5) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 6))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) + 1) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) + 1) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 9) % 6)))) && (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) * 9) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + (((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) + 1) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + (((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 9) / 6)) + 1) % 7) * 28)) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) * 9) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 7))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 7) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 7) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 1) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 7) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 7) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 1) % 6)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 126) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 9)) + 8))] = (((((1 <= (((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 8) / 6)) % 7))) && ((((((int)blockIdx.y) * 7) + ry_outer) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 8) / 6)) % 7)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.x) * 9) + 2) % 6)) < 29)) ? data[((((((((((rc_outer * 9408) + (((int)threadIdx.z) * 2352)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 8) / 6)) / 7) * 784)) + (((int)blockIdx.y) * 196)) + (ry_outer * 28)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 9) + 8) / 6)) % 7) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.x) * 9) + 2) % 6)) - 29))] : 0.000000e+00f);
      kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) / 12) * 864)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) % 12) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 1))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) / 12) * 864)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) % 12) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 2))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) / 12) * 864)) + (rc_outer * 108)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) % 12) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 3))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) % 12) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 4))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) % 12) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 5))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 1) % 12) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 6))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) % 12) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 7))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) % 12) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 8))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 2) % 12) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 9))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) % 12) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 10))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) % 12) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 11))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 3) % 12) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 12))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) % 12) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 13))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) % 12) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 14))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 4) % 12) * 9)) + (ry_outer * 3)) + 2))];
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 379) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1137) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 273) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 15))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) % 12) * 9)) + (ry_outer * 3)))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 379) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1136) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 272) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 16))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) % 12) * 9)) + (ry_outer * 3)) + 1))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 379) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1135) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 271) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 17))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 5) % 12) * 9)) + (ry_outer * 3)) + 2))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 378) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1134) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 270) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 18))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) % 12) * 9)) + (ry_outer * 3)))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 378) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1133) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 269) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 19))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) % 12) * 9)) + (ry_outer * 3)) + 1))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 32) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 7)) < 378) {
          if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) < 1132) {
            if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 21)) < 268) {
              if ((((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 8)) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12)) < 64) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 21)) + 20))] = kernel[((((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 6912)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) / 12) * 864)) + (rc_outer * 108)) + (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 7)) + 6) % 12) * 9)) + (ry_outer * 3)) + 2))];
              }
            }
          }
        }
      }
      __syncthreads();
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 6) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 42))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 44))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 84))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 86))];
      kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 36))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 144))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 288))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 432))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 576))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 720))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 864))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1008))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 3))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 147))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 291))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 435))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 579))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 723))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 867))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1011))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 6))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 150))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 294))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 438))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 582))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 726))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 870))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1014))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 1))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 3))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 43))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 45))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 85))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 87))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 145))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 289))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 433))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 577))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 721))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 865))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1009))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 4))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 148))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 292))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 436))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 580))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 724))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 868))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1012))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 7))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 151))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 295))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 439))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 583))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 727))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 871))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1015))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 4))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 44))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 46))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 86))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 88))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 2))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 146))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 290))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 434))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 578))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 722))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 866))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1010))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 5))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 149))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 293))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 437))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 581))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 725))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 869))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1013))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 8))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 152))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 296))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 440))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 584))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 728))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 872))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1016))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 126))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 128))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 168))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 170))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 210))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 212))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 9))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 153))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 297))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 441))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 585))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 729))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 873))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1017))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 12))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 156))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 300))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 444))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 588))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 732))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 876))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1020))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 15))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 159))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 303))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 447))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 591))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 735))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 879))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1023))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 127))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 129))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 169))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 171))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 211))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 213))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 10))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 154))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 298))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 442))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 586))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 730))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 874))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1018))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 13))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 157))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 301))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 445))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 589))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 733))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 877))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1021))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 16))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 160))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 304))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 448))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 592))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 736))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 880))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1024))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 128))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 130))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 170))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 172))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 212))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 214))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 11))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 155))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 299))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 443))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 587))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 731))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 875))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1019))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 14))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 158))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 302))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 446))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 590))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 734))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 878))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1022))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 17))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 161))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 305))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 449))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 593))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 737))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 881))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1025))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 252))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 254))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 294))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 296))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 336))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 338))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 18))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 162))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 306))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 450))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 594))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 738))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 882))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1026))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 21))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 165))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 309))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 453))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 597))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 741))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 885))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1029))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 24))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 168))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 312))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 456))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 600))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 744))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 888))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1032))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 253))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 255))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 295))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 297))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 337))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 339))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 19))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 163))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 307))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 451))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 595))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 739))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 883))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1027))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 22))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 166))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 310))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 454))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 598))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 742))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 886))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1030))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 25))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 169))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 313))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 457))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 601))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 745))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 889))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1033))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 254))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 256))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 296))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 298))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 338))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 340))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 20))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 164))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 308))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 452))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 596))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 740))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 884))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1028))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 23))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 167))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 311))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 455))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 599))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 743))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 887))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1031))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 26))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 170))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 314))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 458))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 602))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 746))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 890))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1034))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 378))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 380))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 420))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 422))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 462))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 464))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 27))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 171))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 315))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 459))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 603))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 747))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 891))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1035))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 30))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 174))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 318))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 462))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 606))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 750))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 894))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1038))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 33))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 177))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 321))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 465))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 609))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 753))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 897))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1041))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 379))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 381))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 421))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 423))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 463))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 465))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 28))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 172))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 316))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 460))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 604))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 748))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 892))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1036))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 31))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 175))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 319))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 463))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 607))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 751))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 895))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1039))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 34))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 178))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 322))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 466))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 610))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 754))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 898))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1042))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 380))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 382))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 422))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 424))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 464))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 466))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 29))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 173))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 317))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 461))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 605))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 749))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 36) + 893))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1037))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 32))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 176))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 320))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 464))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 608))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 752))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 36) + 896))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1040))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 35))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 179))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 323))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 467))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 611))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 755))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 36) + 899))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1043))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(9)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(15)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(18)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(21)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(18)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(10)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(16)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(19)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(22)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(19)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
      compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(11)]));
      compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(17)]));
      compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(20)]));
      compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(23)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
      compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
      compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
      compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(20)]));
      compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(23)]));
    }
  }
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 3136))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 6272))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 9408))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 12544))] = compute_local[(8)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 15680))] = compute_local[(10)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 18816))] = compute_local[(12)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 21952))] = compute_local[(14)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 2))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 3138))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 6274))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 9410))] = compute_local[(7)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 12546))] = compute_local[(9)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 15682))] = compute_local[(11)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 18818))] = compute_local[(13)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 21954))] = compute_local[(15)];
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
		} 
		break;
		case 2: 
 		switch(write_w){
			case 1:
 			#pragma unroll
			for (unsigned int th = 0; th < 2; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 1; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 2:
 			#pragma unroll
			for (unsigned int th = 0; th < 2; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 2; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
		} 
		break;
		case 3: 
 		switch(write_w){
			case 1:
 			#pragma unroll
			for (unsigned int th = 0; th < 3; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 1; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 2:
 			#pragma unroll
			for (unsigned int th = 0; th < 3; ++th) { 
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
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[8];

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


        dim3 grid(7,4,2);

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
    outfile.open("../../evaluation_outcome/2080Ti-layers-eval-oracle.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_cudnn_host, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<
                                   cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<","<<difference<<endl;
    return 0;
}


