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
#define TH 9
#define TW 2
#define TC 8
#define C 32
#define N 32
#define H 28
#define W 28

#define TCS ((C-1)/TC + 1)
#define THS ((H-1)/TH + 1)
#define TWS ((W-1)/TW+1)
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
  __shared__ float pad_temp_shared[768];
  __shared__ float kernel_shared[72];
  float pad_temp_shared_local[24];
  float kernel_shared_local[18];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 28) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 28) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 28) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 28) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + (((((int)threadIdx.x) * 28) / 96) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 28) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 28) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 1) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 1) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 1) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 1) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 1) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 1) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 2) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 2) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 2) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 2) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 2) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 2) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 3) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 3) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 3) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 3) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 3) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 3) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 4) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 4) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 4) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 4) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 4) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 4) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 5) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 5) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 5) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 5) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 5) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 5) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 6) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 6) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 6) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 6) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 6) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 6) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 7) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 7) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 7) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 7) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 7) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 7) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 8) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 8) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 8) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 8) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 8) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 8) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 9) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 9) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 9) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 9) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 9) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 9) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 10) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 10) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 10) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 10) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 10) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 10) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 11) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 11) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 11) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 11) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 11) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 11) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 12) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 12) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 12) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 12) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 12) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 12) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 12) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 13) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 13) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 13) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 13) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 13) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 13) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 13) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 14) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 14) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 14) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 14) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 14) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 14) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 14) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 15) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 15) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 15) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 15) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 15) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 15) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 15) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 16) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 16) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 28) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 28) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 16) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 16) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 28) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 17) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 17) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 1) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 17) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 17) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 1) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 18) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 18) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 2) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 18) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 18) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 2) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 19) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 19) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 3) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 19) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 19) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 3) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 20) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 20) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 4) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 20) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 20) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 4) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 21))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 21) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 21) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 5) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 21) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 21) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 5) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 22))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 22) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 22) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 6) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 22) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 22) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 6) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 23))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 23) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 23) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 7) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 23) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 23) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 7) & 15)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 28) + 24) / 96)) < 8) {
      if (((((int)threadIdx.y) * 12) + (((((int)threadIdx.x) * 28) + 24) >> 4)) < 48) {
        if (((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) < 744) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 24))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 24) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 24) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 8) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 24) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 24) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 8) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 28) + 25) / 96)) < 8) {
      if (((((int)threadIdx.y) * 12) + (((((int)threadIdx.x) * 28) + 25) >> 4)) < 48) {
        if (((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) < 743) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 25))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 25) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 25) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 9) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 25) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 25) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 9) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 28) + 26) / 96)) < 8) {
      if (((((int)threadIdx.y) * 12) + (((((int)threadIdx.x) * 28) + 26) >> 4)) < 48) {
        if (((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) < 742) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 26))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 26) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 26) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 10) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 26) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 26) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 10) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 28) + 27) / 96)) < 8) {
      if (((((int)threadIdx.y) * 12) + (((((int)threadIdx.x) * 28) + 27) >> 4)) < 48) {
        if (((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) < 741) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.y) * 192) + (((int)threadIdx.x) * 28)) + 27))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 27) % 96) >> 4))) && (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 28) + 27) % 96) >> 4)) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 28) + 11) & 15)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.y) * 1568)) + ((((((int)threadIdx.x) * 28) + 27) / 96) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 28) + 27) % 96) >> 4) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 28) + 11) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) < 8) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 24) {
        if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 72) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[(((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)))] = kernel[(((((((int)blockIdx.z) * 288) + (rc_outer * 72)) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)))];
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) < 8) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 24) {
        if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 71) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[((((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) + 1))] = kernel[((((((((int)blockIdx.z) * 288) + (rc_outer * 72)) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) < 8) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 24) {
        if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 70) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[((((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) + 2))] = kernel[((((((((int)blockIdx.z) * 288) + (rc_outer * 72)) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 3))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 16))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 17))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 18))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 19))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 32))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 33))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 34))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 35))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 96))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 97))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 98))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 99))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 112))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 113))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 114))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 115))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 128))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 129))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 130))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 131))];
    kernel_shared_local[(0)] = kernel_shared[(0)];
    kernel_shared_local[(1)] = kernel_shared[(1)];
    kernel_shared_local[(2)] = kernel_shared[(2)];
    kernel_shared_local[(3)] = kernel_shared[(3)];
    kernel_shared_local[(4)] = kernel_shared[(4)];
    kernel_shared_local[(5)] = kernel_shared[(5)];
    kernel_shared_local[(6)] = kernel_shared[(6)];
    kernel_shared_local[(7)] = kernel_shared[(7)];
    kernel_shared_local[(8)] = kernel_shared[(8)];
    kernel_shared_local[(9)] = kernel_shared[(9)];
    kernel_shared_local[(10)] = kernel_shared[(10)];
    kernel_shared_local[(11)] = kernel_shared[(11)];
    kernel_shared_local[(12)] = kernel_shared[(12)];
    kernel_shared_local[(13)] = kernel_shared[(13)];
    kernel_shared_local[(14)] = kernel_shared[(14)];
    kernel_shared_local[(15)] = kernel_shared[(15)];
    kernel_shared_local[(16)] = kernel_shared[(16)];
    kernel_shared_local[(17)] = kernel_shared[(17)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 192))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 193))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 194))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 195))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 208))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 209))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 210))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 211))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 224))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 225))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 226))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 227))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 288))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 289))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 290))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 291))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 304))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 305))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 306))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 307))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 320))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 321))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 322))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 323))];
    kernel_shared_local[(0)] = kernel_shared[(18)];
    kernel_shared_local[(1)] = kernel_shared[(19)];
    kernel_shared_local[(2)] = kernel_shared[(20)];
    kernel_shared_local[(3)] = kernel_shared[(21)];
    kernel_shared_local[(4)] = kernel_shared[(22)];
    kernel_shared_local[(5)] = kernel_shared[(23)];
    kernel_shared_local[(6)] = kernel_shared[(24)];
    kernel_shared_local[(7)] = kernel_shared[(25)];
    kernel_shared_local[(8)] = kernel_shared[(26)];
    kernel_shared_local[(9)] = kernel_shared[(27)];
    kernel_shared_local[(10)] = kernel_shared[(28)];
    kernel_shared_local[(11)] = kernel_shared[(29)];
    kernel_shared_local[(12)] = kernel_shared[(30)];
    kernel_shared_local[(13)] = kernel_shared[(31)];
    kernel_shared_local[(14)] = kernel_shared[(32)];
    kernel_shared_local[(15)] = kernel_shared[(33)];
    kernel_shared_local[(16)] = kernel_shared[(34)];
    kernel_shared_local[(17)] = kernel_shared[(35)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 384))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 385))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 386))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 387))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 400))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 401))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 402))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 403))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 416))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 417))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 418))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 419))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 480))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 481))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 482))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 483))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 496))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 497))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 498))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 499))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 512))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 513))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 514))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 515))];
    kernel_shared_local[(0)] = kernel_shared[(36)];
    kernel_shared_local[(1)] = kernel_shared[(37)];
    kernel_shared_local[(2)] = kernel_shared[(38)];
    kernel_shared_local[(3)] = kernel_shared[(39)];
    kernel_shared_local[(4)] = kernel_shared[(40)];
    kernel_shared_local[(5)] = kernel_shared[(41)];
    kernel_shared_local[(6)] = kernel_shared[(42)];
    kernel_shared_local[(7)] = kernel_shared[(43)];
    kernel_shared_local[(8)] = kernel_shared[(44)];
    kernel_shared_local[(9)] = kernel_shared[(45)];
    kernel_shared_local[(10)] = kernel_shared[(46)];
    kernel_shared_local[(11)] = kernel_shared[(47)];
    kernel_shared_local[(12)] = kernel_shared[(48)];
    kernel_shared_local[(13)] = kernel_shared[(49)];
    kernel_shared_local[(14)] = kernel_shared[(50)];
    kernel_shared_local[(15)] = kernel_shared[(51)];
    kernel_shared_local[(16)] = kernel_shared[(52)];
    kernel_shared_local[(17)] = kernel_shared[(53)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 576))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 577))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 578))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 579))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 592))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 593))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 594))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 595))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 608))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 609))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 610))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 611))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 672))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 673))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 674))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 675))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 688))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 689))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 690))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 691))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 704))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 705))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 706))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 707))];
    kernel_shared_local[(0)] = kernel_shared[(54)];
    kernel_shared_local[(1)] = kernel_shared[(55)];
    kernel_shared_local[(2)] = kernel_shared[(56)];
    kernel_shared_local[(3)] = kernel_shared[(57)];
    kernel_shared_local[(4)] = kernel_shared[(58)];
    kernel_shared_local[(5)] = kernel_shared[(59)];
    kernel_shared_local[(6)] = kernel_shared[(60)];
    kernel_shared_local[(7)] = kernel_shared[(61)];
    kernel_shared_local[(8)] = kernel_shared[(62)];
    kernel_shared_local[(9)] = kernel_shared[(63)];
    kernel_shared_local[(10)] = kernel_shared[(64)];
    kernel_shared_local[(11)] = kernel_shared[(65)];
    kernel_shared_local[(12)] = kernel_shared[(66)];
    kernel_shared_local[(13)] = kernel_shared[(67)];
    kernel_shared_local[(14)] = kernel_shared[(68)];
    kernel_shared_local[(15)] = kernel_shared[(69)];
    kernel_shared_local[(16)] = kernel_shared[(70)];
    kernel_shared_local[(17)] = kernel_shared[(71)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(17)]));
  }
  compute[((((((((int)blockIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 784) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
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
__device__ void load_data_2_register(float *__restrict__ data_array, unsigned int c_index, const float * __restrict__ kernel, unsigned int n_id){
    for(unsigned int r=0;r<R;++r){
        for(unsigned int s=0;s<S;++s){
            data_array[r*S+s] = kernel[c_index*N*9+r*3*N+s*N+n_id];
        }
    }
}
__device__ void switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result){
	switch (switch_condition) {
		case 0:
			#pragma unroll
			for ( int r = 0; r < 1; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*2+(0-s)] += result;
				}
			}
		break;
		case 1:
			#pragma unroll
			for ( int r = 0; r < 1; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*2+(1-s)] += result;
				}
			}
		break;
		case 2:
			#pragma unroll
			for ( int r = 0; r < 1; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*2+(2-s)] += result;
				}
			}
		break;
		case 3:
			#pragma unroll
			for ( int r = 0; r < 1; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(0-r)*2+(3-s)] += result;
				}
			}
		break;
		case 4:
			#pragma unroll
			for ( int r = 0; r < 2; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*2+(0-s)] += result;
				}
			}
		break;
		case 5:
			#pragma unroll
			for ( int r = 0; r < 2; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*2+(1-s)] += result;
				}
			}
		break;
		case 6:
			#pragma unroll
			for ( int r = 0; r < 2; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*2+(2-s)] += result;
				}
			}
		break;
		case 7:
			#pragma unroll
			for ( int r = 0; r < 2; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(1-r)*2+(3-s)] += result;
				}
			}
		break;
		case 8:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(0-s)] += result;
				}
			}
		break;
		case 9:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(1-s)] += result;
				}
			}
		break;
		case 10:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(2-s)] += result;
				}
			}
		break;
		case 11:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(3-s)] += result;
				}
			}
		break;
		case 12:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(0-s)] += result;
				}
			}
		break;
		case 13:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(1-s)] += result;
				}
			}
		break;
		case 14:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(2-s)] += result;
				}
			}
		break;
		case 15:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(3-s)] += result;
				}
			}
		break;
		case 16:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(4-r)*2+(0-s)] += result;
				}
			}
		break;
		case 17:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(4-r)*2+(1-s)] += result;
				}
			}
		break;
		case 18:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(4-r)*2+(2-s)] += result;
				}
			}
		break;
		case 19:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(4-r)*2+(3-s)] += result;
				}
			}
		break;
		case 20:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(5-r)*2+(0-s)] += result;
				}
			}
		break;
		case 21:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(5-r)*2+(1-s)] += result;
				}
			}
		break;
		case 22:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(5-r)*2+(2-s)] += result;
				}
			}
		break;
		case 23:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(5-r)*2+(3-s)] += result;
				}
			}
		break;
		case 24:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(6-r)*2+(0-s)] += result;
				}
			}
		break;
		case 25:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(6-r)*2+(1-s)] += result;
				}
			}
		break;
		case 26:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(6-r)*2+(2-s)] += result;
				}
			}
		break;
		case 27:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(6-r)*2+(3-s)] += result;
				}
			}
		break;
		case 28:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(7-r)*2+(0-s)] += result;
				}
			}
		break;
		case 29:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(7-r)*2+(1-s)] += result;
				}
			}
		break;
		case 30:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(7-r)*2+(2-s)] += result;
				}
			}
		break;
		case 31:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(7-r)*2+(3-s)] += result;
				}
			}
		break;
		case 32:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(8-r)*2+(0-s)] += result;
				}
			}
		break;
		case 33:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(8-r)*2+(1-s)] += result;
				}
			}
		break;
		case 34:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(8-r)*2+(2-s)] += result;
				}
			}
		break;
		case 35:
			#pragma unroll
			for ( int r = 0; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(8-r)*2+(3-s)] += result;
				}
			}
		break;
		case 36:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(9-r)*2+(0-s)] += result;
				}
			}
		break;
		case 37:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(9-r)*2+(1-s)] += result;
				}
			}
		break;
		case 38:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(9-r)*2+(2-s)] += result;
				}
			}
		break;
		case 39:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(9-r)*2+(3-s)] += result;
				}
			}
		break;
		case 40:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(10-r)*2+(0-s)] += result;
				}
			}
		break;
		case 41:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(10-r)*2+(1-s)] += result;
				}
			}
		break;
		case 42:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(10-r)*2+(2-s)] += result;
				}
			}
		break;
		case 43:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(10-r)*2+(3-s)] += result;
				}
			}
		break;

	}
}
__global__ void transform(float *matrix, float *matrix2){
    for(unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;global_id<C*H*W;global_id+=gridDim.x * blockDim.x){
        const float v = matrix[global_id];
        unsigned int c = global_id / (H*W);
        unsigned int hw = global_id % (H*W);
        int h = (hw)/W+1;
        int w = (hw)%W+1;
        int th_start = min(h/TH,THS-1);
        int tw_start = min(w/TW,TWS-1);
        for(int tile_h_id = th_start;tile_h_id>=0;tile_h_id--){
            if((tile_h_id*TH+TH+2)<=h){
                break;
            }
            for(int tile_w_id = tw_start;tile_w_id>=0;tile_w_id--){
                if((tile_w_id*TW+TW+2)<=w){
                    break;
                }
                unsigned int tile_id = tile_h_id * TWS + tile_w_id;
                unsigned int abs_h = h - tile_h_id*TH;
                unsigned int abs_w = w - tile_w_id*TW;
                matrix2[c*THS*TWS*(TH+2)*(TW+2)+tile_id*(TH+2)*(TW+2)+abs_h*(TW+2)+abs_w] = v;
            }
        }
    }
}
__device__ void load_input_2_shared_memory(float *values,float *shared_input,unsigned int warp_id,unsigned int lane_id,
                                           unsigned int tile_id,unsigned int tile_c_id){
    for(unsigned int c_id=warp_id;c_id<TC&&tile_c_id+c_id<C;c_id+=blockDim.x/32){
        for(unsigned int id = lane_id;id<(TH+2)*(TW+2);id+=32){
            shared_input[c_id*(TH+2)*(TW+2)+id] = values[(tile_c_id+c_id)*(THS*TWS)*(TH+2)*(TW+2)+tile_id*(TH+2)*(TW+2)+id];
        }
    }
}
__global__ void conv2d(float * __restrict__ values,const float * __restrict__ kernel, float * __restrict__ outputs){
    __shared__ float input[TC*(TH+2)*(TW+2)];
    const unsigned int tile_id = blockIdx.x;
    const unsigned int tc_id = tile_id / (THS * TWS);
    const unsigned int th_id = (tile_id - tc_id * (THS*TWS))/TWS;
    const unsigned int tw_id = (tile_id - tc_id * (THS*TWS))%TWS;
    const unsigned int h_start = th_id * TH;
    const unsigned int w_start = tw_id * TW;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    float data_array[9];
    float temp_result[TH*TW] = {0.0f};
    load_input_2_shared_memory(values,input,warp_id,lane_id,tile_id - tc_id * (THS*TWS),tc_id*TC);
    __syncthreads();
    float v;
    unsigned int n = threadIdx.x;
    unsigned int c_offset = tc_id * TC;
#pragma unroll
    for(unsigned int c=0;c<TC;c++){
        load_data_2_register(data_array,c + c_offset,kernel,n);
#pragma unroll
        for(unsigned int i=0;i<(TH+2)*(TW+2);++i){
            v = input[i + c*(TH+2)*(TW+2)];
            switch_function(i,data_array,v,temp_result);
        }
    }
#pragma unroll
    for (unsigned int th = 0; th < TH; ++th) {
#pragma unroll
        for (unsigned int tw = 0; tw < TW; ++tw) {
            if (h_start + th >= H || w_start + tw >= W) {
                continue;
            }
            atomicAdd(&outputs[n*H*W+(h_start + th) * W+(w_start + tw)],temp_result[(th * TW + tw)]);
        }
    }
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


        dim3 grid(2,7,32);

        dim3 block(7,4,1);

    cudaEventRecord(event_start);
    default_function_kernel0<<<grid, block>>>(device_input, device_K, device_out);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_tvm;
    cudaEventElapsedTime(&time_tvm, event_start, event_stop);
    float *out_tvm = new float[N*H*W];
    cudaMemcpy(out_tvm,device_out,N*H*W*sizeof(float),cudaMemcpyDeviceToHost);

    cudaMemset(device_out, 0, sizeof(float)*N*H*W);

    unsigned int blkDim = ((N - 1)/32 + 1) * 32;
    cudaEventRecord(event_start);
    transform<<<216,1024>>>(device_input,matrix);
    conv2d<<<TCS*THS*TWS,blkDim>>>(matrix,device_K, device_out);
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
    outfile.open("../../evaluation_outcome/A100-layers-eval-oracle.csv", std::ios_base::app);
    outfile << buffer;



    float difference = check_diff(out_tvm, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<endl;
    return 0;
}


