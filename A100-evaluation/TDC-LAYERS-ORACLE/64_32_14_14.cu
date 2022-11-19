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
#define TH 2
#define TW 2
#define TC 8
#define C 64
#define N 32
#define H 14
#define W 14

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
  float compute_local[4];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[192];
  float pad_temp_shared_local[24];
  float kernel_shared_local[24];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.x) * 74))] = (((((1 <= (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((((int)threadIdx.x) * 74) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 31) >> 4) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 1))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 1) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 2))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 2) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 3))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 3) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 4))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 4) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 5))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 5) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 6))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 6) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 7))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 7) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 8))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 8) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 9))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 9) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 10))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 10) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 10) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 11))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 11) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 11) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 12))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 12) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 12) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 13))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 13) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 13) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 14))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 14) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 14) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 15))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 15) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 15) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 16))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 31) >> 4)) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 16) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 16) & 31) >> 4) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 17))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 17) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 17) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 18))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 18) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 18) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 19))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 19) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 19) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 20))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 20) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 20) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 21))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 21) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 21) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 22))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 22) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 22) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 23))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 23) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 23) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 24))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 24) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 24) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 25))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 25) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 25) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 26))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 26) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 26) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 27))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 27) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 27) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 28))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 28) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 28) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 29))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 29) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 29) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 30))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 30) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 30) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 31))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 31) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 31) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 32))] = (((((1 <= (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((((int)threadIdx.x) * 74) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 31) >> 4) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 181))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 33))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 33) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 34))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 34) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 35))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 35) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 36))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 36) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 37))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 37) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 38))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 38) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 39))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 39) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 40))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 40) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 41))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 41) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 42))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 42) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 10) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 43))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 43) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 11) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 44))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 44) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 12) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 45))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 45) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 13) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 46))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 46) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 14) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 47))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 47) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 15) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 48))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 31) >> 4)) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 48) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 16) & 31) >> 4) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 49))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 49) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 17) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 50))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 50) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 18) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 51))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 51) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 19) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 52))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 52) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 20) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 53))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 53) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 21) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 54))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 54) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 22) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 55))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 55) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 23) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 56))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 56) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 24) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 57))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 57) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 25) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 58))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 58) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 26) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 59))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 59) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 27) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 60))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 60) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 28) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 61))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 61) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 29) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 62))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 62) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 30) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 63))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 63) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 31) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 64))] = (((((1 <= (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 31) >> 4)) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((((int)threadIdx.x) * 74) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 31) >> 4) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 377))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 65))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 65) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 66))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 66) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 67))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 67) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 68))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 68) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 69))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 69) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 70))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 70) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 71))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 71) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 72))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 72) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 73))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 31) >> 4)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.x) * 74) + 73) >> 5) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 31) >> 4) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      }
      kernel_shared[((((int)threadIdx.x) * 14))] = kernel[(((((((((int)blockIdx.z) * 1152) + (((((int)threadIdx.x) * 14) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) % 96) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 14) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 1))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 1) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 1) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 2))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 2) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 2) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 3))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 3) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 3) % 96) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 14) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 4))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 4) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 4) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 5))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 5) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 5) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 6))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 6) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 6) % 96) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 14) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 7))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 7) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 7) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 8))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 8) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 8) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 14) + 9))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 9) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 9) % 96) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 14) % 3)))];
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 14) + 10))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 10) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 10) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 1) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 14) + 11))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 11) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 11) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 2) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 14) + 12))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 12) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 12) % 96) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 14) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 14) + 13))] = kernel[(((((((((int)blockIdx.z) * 1152) + ((((((int)threadIdx.x) * 14) + 13) / 96) * 576)) + (rc_outer * 288)) + (((((((int)threadIdx.x) * 14) + 13) % 96) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 14) + 1) % 3)))];
      }
      __syncthreads();
      pad_temp_shared_local[(0)] = pad_temp_shared[(((int)threadIdx.x))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 1))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 16))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 17))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 18))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 32))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 33))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 34))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 48))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 49))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 50))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 64))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 65))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 66))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 80))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 81))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 82))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 96))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 97))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 98))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 112))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 113))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 114))];
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
      kernel_shared_local[(12)] = kernel_shared[(96)];
      kernel_shared_local[(13)] = kernel_shared[(97)];
      kernel_shared_local[(14)] = kernel_shared[(98)];
      kernel_shared_local[(15)] = kernel_shared[(99)];
      kernel_shared_local[(16)] = kernel_shared[(100)];
      kernel_shared_local[(17)] = kernel_shared[(101)];
      kernel_shared_local[(18)] = kernel_shared[(102)];
      kernel_shared_local[(19)] = kernel_shared[(103)];
      kernel_shared_local[(20)] = kernel_shared[(104)];
      kernel_shared_local[(21)] = kernel_shared[(105)];
      kernel_shared_local[(22)] = kernel_shared[(106)];
      kernel_shared_local[(23)] = kernel_shared[(107)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 128))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 129))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 130))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 144))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 145))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 146))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 160))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 161))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 162))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 176))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 177))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 178))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 192))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 193))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 194))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 208))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 209))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 210))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 224))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 225))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 226))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 240))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 241))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 242))];
      kernel_shared_local[(0)] = kernel_shared[(12)];
      kernel_shared_local[(1)] = kernel_shared[(13)];
      kernel_shared_local[(2)] = kernel_shared[(14)];
      kernel_shared_local[(3)] = kernel_shared[(15)];
      kernel_shared_local[(4)] = kernel_shared[(16)];
      kernel_shared_local[(5)] = kernel_shared[(17)];
      kernel_shared_local[(6)] = kernel_shared[(18)];
      kernel_shared_local[(7)] = kernel_shared[(19)];
      kernel_shared_local[(8)] = kernel_shared[(20)];
      kernel_shared_local[(9)] = kernel_shared[(21)];
      kernel_shared_local[(10)] = kernel_shared[(22)];
      kernel_shared_local[(11)] = kernel_shared[(23)];
      kernel_shared_local[(12)] = kernel_shared[(108)];
      kernel_shared_local[(13)] = kernel_shared[(109)];
      kernel_shared_local[(14)] = kernel_shared[(110)];
      kernel_shared_local[(15)] = kernel_shared[(111)];
      kernel_shared_local[(16)] = kernel_shared[(112)];
      kernel_shared_local[(17)] = kernel_shared[(113)];
      kernel_shared_local[(18)] = kernel_shared[(114)];
      kernel_shared_local[(19)] = kernel_shared[(115)];
      kernel_shared_local[(20)] = kernel_shared[(116)];
      kernel_shared_local[(21)] = kernel_shared[(117)];
      kernel_shared_local[(22)] = kernel_shared[(118)];
      kernel_shared_local[(23)] = kernel_shared[(119)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 256))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 257))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 258))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 272))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 273))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 274))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 288))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 289))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 290))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 304))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 305))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 306))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 320))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 321))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 322))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 336))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 337))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 338))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 352))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 353))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 354))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 368))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 369))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 370))];
      kernel_shared_local[(0)] = kernel_shared[(24)];
      kernel_shared_local[(1)] = kernel_shared[(25)];
      kernel_shared_local[(2)] = kernel_shared[(26)];
      kernel_shared_local[(3)] = kernel_shared[(27)];
      kernel_shared_local[(4)] = kernel_shared[(28)];
      kernel_shared_local[(5)] = kernel_shared[(29)];
      kernel_shared_local[(6)] = kernel_shared[(30)];
      kernel_shared_local[(7)] = kernel_shared[(31)];
      kernel_shared_local[(8)] = kernel_shared[(32)];
      kernel_shared_local[(9)] = kernel_shared[(33)];
      kernel_shared_local[(10)] = kernel_shared[(34)];
      kernel_shared_local[(11)] = kernel_shared[(35)];
      kernel_shared_local[(12)] = kernel_shared[(120)];
      kernel_shared_local[(13)] = kernel_shared[(121)];
      kernel_shared_local[(14)] = kernel_shared[(122)];
      kernel_shared_local[(15)] = kernel_shared[(123)];
      kernel_shared_local[(16)] = kernel_shared[(124)];
      kernel_shared_local[(17)] = kernel_shared[(125)];
      kernel_shared_local[(18)] = kernel_shared[(126)];
      kernel_shared_local[(19)] = kernel_shared[(127)];
      kernel_shared_local[(20)] = kernel_shared[(128)];
      kernel_shared_local[(21)] = kernel_shared[(129)];
      kernel_shared_local[(22)] = kernel_shared[(130)];
      kernel_shared_local[(23)] = kernel_shared[(131)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 384))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 385))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 386))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 400))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 401))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 402))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 416))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 417))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 418))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 432))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 433))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 434))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 448))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 449))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 450))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 464))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 465))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 466))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 480))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 481))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 482))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 496))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 497))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 498))];
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
      kernel_shared_local[(12)] = kernel_shared[(132)];
      kernel_shared_local[(13)] = kernel_shared[(133)];
      kernel_shared_local[(14)] = kernel_shared[(134)];
      kernel_shared_local[(15)] = kernel_shared[(135)];
      kernel_shared_local[(16)] = kernel_shared[(136)];
      kernel_shared_local[(17)] = kernel_shared[(137)];
      kernel_shared_local[(18)] = kernel_shared[(138)];
      kernel_shared_local[(19)] = kernel_shared[(139)];
      kernel_shared_local[(20)] = kernel_shared[(140)];
      kernel_shared_local[(21)] = kernel_shared[(141)];
      kernel_shared_local[(22)] = kernel_shared[(142)];
      kernel_shared_local[(23)] = kernel_shared[(143)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 512))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 513))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 514))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 528))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 529))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 530))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 544))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 545))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 546))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 560))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 561))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 562))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 576))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 577))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 578))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 592))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 593))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 594))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 608))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 609))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 610))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 624))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 625))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 626))];
      kernel_shared_local[(0)] = kernel_shared[(48)];
      kernel_shared_local[(1)] = kernel_shared[(49)];
      kernel_shared_local[(2)] = kernel_shared[(50)];
      kernel_shared_local[(3)] = kernel_shared[(51)];
      kernel_shared_local[(4)] = kernel_shared[(52)];
      kernel_shared_local[(5)] = kernel_shared[(53)];
      kernel_shared_local[(6)] = kernel_shared[(54)];
      kernel_shared_local[(7)] = kernel_shared[(55)];
      kernel_shared_local[(8)] = kernel_shared[(56)];
      kernel_shared_local[(9)] = kernel_shared[(57)];
      kernel_shared_local[(10)] = kernel_shared[(58)];
      kernel_shared_local[(11)] = kernel_shared[(59)];
      kernel_shared_local[(12)] = kernel_shared[(144)];
      kernel_shared_local[(13)] = kernel_shared[(145)];
      kernel_shared_local[(14)] = kernel_shared[(146)];
      kernel_shared_local[(15)] = kernel_shared[(147)];
      kernel_shared_local[(16)] = kernel_shared[(148)];
      kernel_shared_local[(17)] = kernel_shared[(149)];
      kernel_shared_local[(18)] = kernel_shared[(150)];
      kernel_shared_local[(19)] = kernel_shared[(151)];
      kernel_shared_local[(20)] = kernel_shared[(152)];
      kernel_shared_local[(21)] = kernel_shared[(153)];
      kernel_shared_local[(22)] = kernel_shared[(154)];
      kernel_shared_local[(23)] = kernel_shared[(155)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 640))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 641))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 642))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 656))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 657))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 658))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 672))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 673))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 674))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 688))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 689))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 690))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 704))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 705))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 706))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 720))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 721))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 722))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 736))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 737))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 738))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 752))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 753))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 754))];
      kernel_shared_local[(0)] = kernel_shared[(60)];
      kernel_shared_local[(1)] = kernel_shared[(61)];
      kernel_shared_local[(2)] = kernel_shared[(62)];
      kernel_shared_local[(3)] = kernel_shared[(63)];
      kernel_shared_local[(4)] = kernel_shared[(64)];
      kernel_shared_local[(5)] = kernel_shared[(65)];
      kernel_shared_local[(6)] = kernel_shared[(66)];
      kernel_shared_local[(7)] = kernel_shared[(67)];
      kernel_shared_local[(8)] = kernel_shared[(68)];
      kernel_shared_local[(9)] = kernel_shared[(69)];
      kernel_shared_local[(10)] = kernel_shared[(70)];
      kernel_shared_local[(11)] = kernel_shared[(71)];
      kernel_shared_local[(12)] = kernel_shared[(156)];
      kernel_shared_local[(13)] = kernel_shared[(157)];
      kernel_shared_local[(14)] = kernel_shared[(158)];
      kernel_shared_local[(15)] = kernel_shared[(159)];
      kernel_shared_local[(16)] = kernel_shared[(160)];
      kernel_shared_local[(17)] = kernel_shared[(161)];
      kernel_shared_local[(18)] = kernel_shared[(162)];
      kernel_shared_local[(19)] = kernel_shared[(163)];
      kernel_shared_local[(20)] = kernel_shared[(164)];
      kernel_shared_local[(21)] = kernel_shared[(165)];
      kernel_shared_local[(22)] = kernel_shared[(166)];
      kernel_shared_local[(23)] = kernel_shared[(167)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 768))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 769))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 770))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 784))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 785))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 786))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 800))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 801))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 802))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 816))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 817))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 818))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 832))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 833))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 834))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 848))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 849))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 850))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 864))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 865))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 866))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 880))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 881))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 882))];
      kernel_shared_local[(0)] = kernel_shared[(72)];
      kernel_shared_local[(1)] = kernel_shared[(73)];
      kernel_shared_local[(2)] = kernel_shared[(74)];
      kernel_shared_local[(3)] = kernel_shared[(75)];
      kernel_shared_local[(4)] = kernel_shared[(76)];
      kernel_shared_local[(5)] = kernel_shared[(77)];
      kernel_shared_local[(6)] = kernel_shared[(78)];
      kernel_shared_local[(7)] = kernel_shared[(79)];
      kernel_shared_local[(8)] = kernel_shared[(80)];
      kernel_shared_local[(9)] = kernel_shared[(81)];
      kernel_shared_local[(10)] = kernel_shared[(82)];
      kernel_shared_local[(11)] = kernel_shared[(83)];
      kernel_shared_local[(12)] = kernel_shared[(168)];
      kernel_shared_local[(13)] = kernel_shared[(169)];
      kernel_shared_local[(14)] = kernel_shared[(170)];
      kernel_shared_local[(15)] = kernel_shared[(171)];
      kernel_shared_local[(16)] = kernel_shared[(172)];
      kernel_shared_local[(17)] = kernel_shared[(173)];
      kernel_shared_local[(18)] = kernel_shared[(174)];
      kernel_shared_local[(19)] = kernel_shared[(175)];
      kernel_shared_local[(20)] = kernel_shared[(176)];
      kernel_shared_local[(21)] = kernel_shared[(177)];
      kernel_shared_local[(22)] = kernel_shared[(178)];
      kernel_shared_local[(23)] = kernel_shared[(179)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 896))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 897))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 898))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 912))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 913))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 914))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 928))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 929))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 930))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 944))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 945))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 946))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 960))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 961))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 962))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 976))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.x) + 977))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.x) + 978))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.x) + 992))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.x) + 993))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.x) + 994))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.x) + 1008))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.x) + 1009))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.x) + 1010))];
      kernel_shared_local[(0)] = kernel_shared[(84)];
      kernel_shared_local[(1)] = kernel_shared[(85)];
      kernel_shared_local[(2)] = kernel_shared[(86)];
      kernel_shared_local[(3)] = kernel_shared[(87)];
      kernel_shared_local[(4)] = kernel_shared[(88)];
      kernel_shared_local[(5)] = kernel_shared[(89)];
      kernel_shared_local[(6)] = kernel_shared[(90)];
      kernel_shared_local[(7)] = kernel_shared[(91)];
      kernel_shared_local[(8)] = kernel_shared[(92)];
      kernel_shared_local[(9)] = kernel_shared[(93)];
      kernel_shared_local[(10)] = kernel_shared[(94)];
      kernel_shared_local[(11)] = kernel_shared[(95)];
      kernel_shared_local[(12)] = kernel_shared[(180)];
      kernel_shared_local[(13)] = kernel_shared[(181)];
      kernel_shared_local[(14)] = kernel_shared[(182)];
      kernel_shared_local[(15)] = kernel_shared[(183)];
      kernel_shared_local[(16)] = kernel_shared[(184)];
      kernel_shared_local[(17)] = kernel_shared[(185)];
      kernel_shared_local[(18)] = kernel_shared[(186)];
      kernel_shared_local[(19)] = kernel_shared[(187)];
      kernel_shared_local[(20)] = kernel_shared[(188)];
      kernel_shared_local[(21)] = kernel_shared[(189)];
      kernel_shared_local[(22)] = kernel_shared[(190)];
      kernel_shared_local[(23)] = kernel_shared[(191)];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
    }
  }
  compute[((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
  compute[(((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 196))] = compute_local[(2)];
  compute[(((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 210))] = compute_local[(3)];
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
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(0-s)] += result;
				}
			}
		break;
		case 9:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(1-s)] += result;
				}
			}
		break;
		case 10:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(2-s)] += result;
				}
			}
		break;
		case 11:
			#pragma unroll
			for ( int r = 1; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(2-r)*2+(3-s)] += result;
				}
			}
		break;
		case 12:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 1; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(0-s)] += result;
				}
			}
		break;
		case 13:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 0; s < 2; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(1-s)] += result;
				}
			}
		break;
		case 14:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 1; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(2-s)] += result;
				}
			}
		break;
		case 15:
			#pragma unroll
			for ( int r = 2; r < 3; r++) {
				#pragma unroll
				for ( int s = 2; s < 3; s++) {
					float result = v * temp_kernel[r*S+s];
					temp_result[(3-r)*2+(3-s)] += result;
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


        dim3 grid(1,7,16);

        dim3 block(14,1,1);

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


