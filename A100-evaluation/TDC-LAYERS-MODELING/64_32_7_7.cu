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
#define TW 1
#define TC 16
#define C 64
#define N 32
#define H 7
#define W 7

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
  float compute_local[1];
  __shared__ float pad_temp_shared[864];
  __shared__ float kernel_shared[1152];
  float pad_temp_shared_local[24];
  float kernel_shared_local[24];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)))] = (((((1 <= ((((((int)threadIdx.x) * 31) % 27) / 9) + ((int)blockIdx.y))) && (((((((int)threadIdx.x) * 31) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= ((((int)threadIdx.x) * 31) % 9))) && (((((int)threadIdx.x) * 31) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((int)threadIdx.x) * 31) / 27) * 49)) + ((((((int)threadIdx.x) * 31) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) * 31) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 1))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 1) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 1) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 1) % 9))) && ((((((int)threadIdx.x) * 31) + 1) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 1) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 1) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 1) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 2))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 2) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 2) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 2) % 9))) && ((((((int)threadIdx.x) * 31) + 2) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 2) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 2) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 2) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 3))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 3) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 3) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 3) % 9))) && ((((((int)threadIdx.x) * 31) + 3) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 3) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 3) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 3) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 4))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 4) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 4) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 4) % 9))) && ((((((int)threadIdx.x) * 31) + 4) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 4) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 4) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 4) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 5))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 5) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 5) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 5) % 9))) && ((((((int)threadIdx.x) * 31) + 5) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 5) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 5) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 5) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 6))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 6) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 6) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 6) % 9))) && ((((((int)threadIdx.x) * 31) + 6) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 6) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 6) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 6) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 7))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 7) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 7) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 7) % 9))) && ((((((int)threadIdx.x) * 31) + 7) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 7) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 7) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 7) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 8))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 8) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 8) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 8) % 9))) && ((((((int)threadIdx.x) * 31) + 8) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 8) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 8) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 8) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 9))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 9) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 9) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= ((((int)threadIdx.x) * 31) % 9))) && (((((int)threadIdx.x) * 31) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 9) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 9) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) * 31) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 10))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 10) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 10) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 1) % 9))) && ((((((int)threadIdx.x) * 31) + 1) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 10) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 10) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 1) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 11))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 11) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 11) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 2) % 9))) && ((((((int)threadIdx.x) * 31) + 2) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 11) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 11) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 2) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 12))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 12) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 12) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 3) % 9))) && ((((((int)threadIdx.x) * 31) + 3) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 12) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 12) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 3) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 13))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 13) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 13) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 4) % 9))) && ((((((int)threadIdx.x) * 31) + 4) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 13) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 13) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 4) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 14))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 14) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 14) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 5) % 9))) && ((((((int)threadIdx.x) * 31) + 5) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 14) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 14) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 5) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 15))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 15) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 15) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 6) % 9))) && ((((((int)threadIdx.x) * 31) + 6) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 15) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 15) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 6) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 16))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 16) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 16) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 7) % 9))) && ((((((int)threadIdx.x) * 31) + 7) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 16) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 16) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 7) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 17))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 17) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 17) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 8) % 9))) && ((((((int)threadIdx.x) * 31) + 8) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 17) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 17) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 8) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 18))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 18) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 18) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= ((((int)threadIdx.x) * 31) % 9))) && (((((int)threadIdx.x) * 31) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 18) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 18) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) * 31) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 19))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 19) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 19) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 1) % 9))) && ((((((int)threadIdx.x) * 31) + 1) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 19) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 19) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 1) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 20))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 20) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 20) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 2) % 9))) && ((((((int)threadIdx.x) * 31) + 2) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 20) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 20) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 2) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 21))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 21) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 21) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 3) % 9))) && ((((((int)threadIdx.x) * 31) + 3) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 21) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 21) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 3) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 22))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 22) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 22) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 4) % 9))) && ((((((int)threadIdx.x) * 31) + 4) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 22) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 22) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 4) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 23))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 23) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 23) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 5) % 9))) && ((((((int)threadIdx.x) * 31) + 5) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 23) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 23) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 5) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 24))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 24) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 24) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 6) % 9))) && ((((((int)threadIdx.x) * 31) + 6) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 24) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 24) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 6) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 25))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 25) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 25) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 7) % 9))) && ((((((int)threadIdx.x) * 31) + 7) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 25) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 25) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 7) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 26))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 26) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 26) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 8) % 9))) && ((((((int)threadIdx.x) * 31) + 8) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 26) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 26) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 8) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 27))] = (((((1 <= ((((((int)threadIdx.x) * 31) % 27) / 9) + ((int)blockIdx.y))) && (((((((int)threadIdx.x) * 31) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= ((((int)threadIdx.x) * 31) % 9))) && (((((int)threadIdx.x) * 31) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((int)threadIdx.x) * 31) / 27) * 49)) + ((((((int)threadIdx.x) * 31) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) * 31) % 9)) + 41))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 28))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 1) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 1) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 1) % 9))) && ((((((int)threadIdx.x) * 31) + 1) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 28) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 1) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 1) % 9)) - 8))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 29))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 2) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 2) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 2) % 9))) && ((((((int)threadIdx.x) * 31) + 2) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 29) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 2) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 2) % 9)) - 8))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 31) + 30) / 27)) < 32) {
      if (((((int)threadIdx.z) * 24) + (((((int)threadIdx.x) * 31) + 30) / 9)) < 96) {
        if (((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) < 834) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.x) * 31)) + 30))] = (((((1 <= (((((((int)threadIdx.x) * 31) + 3) % 27) / 9) + ((int)blockIdx.y))) && ((((((((int)threadIdx.x) * 31) + 3) % 27) / 9) + ((int)blockIdx.y)) < 8)) && (1 <= (((((int)threadIdx.x) * 31) + 3) % 9))) && ((((((int)threadIdx.x) * 31) + 3) % 9) < 8)) ? data[((((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.x) * 31) + 30) / 27) * 49)) + (((((((int)threadIdx.x) * 31) + 3) % 27) / 9) * 7)) + (((int)blockIdx.y) * 7)) + (((((int)threadIdx.x) * 31) + 3) % 9)) - 8))] : 0.000000e+00f);
          }
        }
      }
    }
    kernel_shared[(((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)))] = kernel[(((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 1))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 2))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 3))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 3))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 4))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 4))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 5))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 5))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 6))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 6))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 7))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 7))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 8))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 8))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 9))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 9))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 10))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 10))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 11))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 11))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 12))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 12))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 13))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 13))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 14))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 14))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 15))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 15))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 16))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 16))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 17))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 17))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 18))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 18))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 19))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 19))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 20))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 20))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 21))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 21))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 22))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 22))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 23))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 23))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 24))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 24))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 25))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 25))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 26))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 26))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 27))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 27))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 28))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 28))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 29))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 29))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 30))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 30))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 31))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 31))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 32))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 32))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 33))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 33))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 34))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 34))];
    kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 35))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 35))];
    if (((((((int)threadIdx.x) * 14) + 12) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1116) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 36))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 36))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 14) + 12) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1115) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 37))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 37))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 14) + 12) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1114) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 38))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 38))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 14) + 13) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1113) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 39))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 39))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 14) + 13) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1112) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 40))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 40))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 14) + 13) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 96) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) < 1111) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.x) * 42)) + 41))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 576)) + (rc_outer * 288)) + (((int)threadIdx.x) * 42)) + 41))];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 4; ++rc_inner_outer) {
      pad_temp_shared_local[(0)] = pad_temp_shared[(((rc_inner_outer * 216) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 1))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 27))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 28))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 29))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 54))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 55))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 56))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 81))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 82))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 83))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 108))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 109))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 110))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 135))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 136))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 137))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 162))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 163))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 164))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 189))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 190))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 191))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 1))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 2))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 9))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 10))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 11))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 18))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 19))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 20))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 27))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 28))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 29))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 36))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 37))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 38))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 45))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 46))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 47))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 54))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 55))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 56))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 63))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 64))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 65))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 9))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 10))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 11))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 36))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 37))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 38))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 63))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 64))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 65))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 90))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 91))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 92))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 117))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 118))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 119))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 144))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 145))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 146))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 171))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 172))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 173))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 198))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 199))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 200))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 3))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 4))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 5))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 12))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 13))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 14))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 21))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 22))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 23))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 30))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 31))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 32))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 39))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 40))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 41))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 48))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 49))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 50))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 57))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 58))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 59))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 66))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 67))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 68))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 18))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 19))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 20))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 45))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 46))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 47))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 72))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 73))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 74))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 99))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 100))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 101))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 126))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 127))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 128))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 153))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 154))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 155))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 180))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 181))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 182))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 207))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 208))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((rc_inner_outer * 216) + ((int)threadIdx.x)) + 209))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 6))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 7))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 8))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 15))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 16))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 17))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 24))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 25))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 26))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 33))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 34))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 35))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 42))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 43))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 44))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 51))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 52))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 53))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 60))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 61))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 62))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 69))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 70))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 72)) + 71))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
    }
  }
  compute[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
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
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];

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


        dim3 grid(1,7,8);

                dim3 block(7,1,4);

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


