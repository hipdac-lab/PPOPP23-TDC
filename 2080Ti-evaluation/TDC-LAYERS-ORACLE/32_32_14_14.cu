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
#define TC 4
#define C 32
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
  float compute_local[2];
  __shared__ float pad_temp_shared[576];
  __shared__ float kernel_shared[72];
  float pad_temp_shared_local[6];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 576) {
      pad_temp_shared[(((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)))] = (((((9 <= (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 144)) && ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 9)) < 15)) ? data[(((((((rc_outer * 784) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) / 144) * 196)) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 575) {
      pad_temp_shared[((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1))] = (((((9 <= ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 144)) && (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 9)))) && (((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 9)) < 15)) ? data[(((((((rc_outer * 784) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) / 144) * 196)) + ((((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 1) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 574) {
      pad_temp_shared[((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2))] = (((((9 <= ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 144)) && (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 9)))) && (((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 9)) < 15)) ? data[(((((((rc_outer * 784) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) / 144) * 196)) + ((((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 2) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 573) {
      pad_temp_shared[((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3))] = (((((9 <= ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 144)) && (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 9)))) && (((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 9)) < 15)) ? data[(((((((rc_outer * 784) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) / 144) * 196)) + ((((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 3) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 572) {
      pad_temp_shared[((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4))] = (((((9 <= ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 144)) && (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 9)))) && (((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 9)) < 15)) ? data[(((((((rc_outer * 784) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) / 144) * 196)) + ((((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 4) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 571) {
      pad_temp_shared[((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5))] = (((((9 <= ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 144)) && (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 144) < 135)) && (1 <= ((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 9)))) && (((((int)blockIdx.x) * 7) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 9)) < 15)) ? data[(((((((rc_outer * 784) + (((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) / 144) * 196)) + ((((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 144) / 9) * 14)) + (((int)blockIdx.x) * 7)) + ((((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) + 5) % 9)) - 15))] : 0.000000e+00f);
    }
    if (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) < 24) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 72) {
        if (((int)threadIdx.x) < 6) {
          kernel_shared[(((((int)threadIdx.y) * 6) + ((int)threadIdx.x)))] = kernel[((((((((int)blockIdx.z) * 576) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) / 12) * 288)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) % 12) * 3)) + (((int)threadIdx.x) % 3)))];
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 9) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 144))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 153))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 162))];
    kernel_shared_local[(0)] = kernel_shared[(0)];
    kernel_shared_local[(1)] = kernel_shared[(3)];
    kernel_shared_local[(2)] = kernel_shared[(6)];
    kernel_shared_local[(3)] = kernel_shared[(9)];
    kernel_shared_local[(4)] = kernel_shared[(12)];
    kernel_shared_local[(5)] = kernel_shared[(15)];
    kernel_shared_local[(6)] = kernel_shared[(36)];
    kernel_shared_local[(7)] = kernel_shared[(39)];
    kernel_shared_local[(8)] = kernel_shared[(42)];
    kernel_shared_local[(9)] = kernel_shared[(45)];
    kernel_shared_local[(10)] = kernel_shared[(48)];
    kernel_shared_local[(11)] = kernel_shared[(51)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 19))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 145))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 154))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 163))];
    kernel_shared_local[(0)] = kernel_shared[(1)];
    kernel_shared_local[(1)] = kernel_shared[(4)];
    kernel_shared_local[(2)] = kernel_shared[(7)];
    kernel_shared_local[(3)] = kernel_shared[(10)];
    kernel_shared_local[(4)] = kernel_shared[(13)];
    kernel_shared_local[(5)] = kernel_shared[(16)];
    kernel_shared_local[(6)] = kernel_shared[(37)];
    kernel_shared_local[(7)] = kernel_shared[(40)];
    kernel_shared_local[(8)] = kernel_shared[(43)];
    kernel_shared_local[(9)] = kernel_shared[(46)];
    kernel_shared_local[(10)] = kernel_shared[(49)];
    kernel_shared_local[(11)] = kernel_shared[(52)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 146))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 155))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 164))];
    kernel_shared_local[(0)] = kernel_shared[(2)];
    kernel_shared_local[(1)] = kernel_shared[(5)];
    kernel_shared_local[(2)] = kernel_shared[(8)];
    kernel_shared_local[(3)] = kernel_shared[(11)];
    kernel_shared_local[(4)] = kernel_shared[(14)];
    kernel_shared_local[(5)] = kernel_shared[(17)];
    kernel_shared_local[(6)] = kernel_shared[(38)];
    kernel_shared_local[(7)] = kernel_shared[(41)];
    kernel_shared_local[(8)] = kernel_shared[(44)];
    kernel_shared_local[(9)] = kernel_shared[(47)];
    kernel_shared_local[(10)] = kernel_shared[(50)];
    kernel_shared_local[(11)] = kernel_shared[(53)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 288))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 297))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 306))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 432))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 441))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 450))];
    kernel_shared_local[(0)] = kernel_shared[(18)];
    kernel_shared_local[(1)] = kernel_shared[(21)];
    kernel_shared_local[(2)] = kernel_shared[(24)];
    kernel_shared_local[(3)] = kernel_shared[(27)];
    kernel_shared_local[(4)] = kernel_shared[(30)];
    kernel_shared_local[(5)] = kernel_shared[(33)];
    kernel_shared_local[(6)] = kernel_shared[(54)];
    kernel_shared_local[(7)] = kernel_shared[(57)];
    kernel_shared_local[(8)] = kernel_shared[(60)];
    kernel_shared_local[(9)] = kernel_shared[(63)];
    kernel_shared_local[(10)] = kernel_shared[(66)];
    kernel_shared_local[(11)] = kernel_shared[(69)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 289))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 298))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 307))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 433))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 442))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 451))];
    kernel_shared_local[(0)] = kernel_shared[(19)];
    kernel_shared_local[(1)] = kernel_shared[(22)];
    kernel_shared_local[(2)] = kernel_shared[(25)];
    kernel_shared_local[(3)] = kernel_shared[(28)];
    kernel_shared_local[(4)] = kernel_shared[(31)];
    kernel_shared_local[(5)] = kernel_shared[(34)];
    kernel_shared_local[(6)] = kernel_shared[(55)];
    kernel_shared_local[(7)] = kernel_shared[(58)];
    kernel_shared_local[(8)] = kernel_shared[(61)];
    kernel_shared_local[(9)] = kernel_shared[(64)];
    kernel_shared_local[(10)] = kernel_shared[(67)];
    kernel_shared_local[(11)] = kernel_shared[(70)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 290))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 299))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 308))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 434))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 443))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 452))];
    kernel_shared_local[(0)] = kernel_shared[(20)];
    kernel_shared_local[(1)] = kernel_shared[(23)];
    kernel_shared_local[(2)] = kernel_shared[(26)];
    kernel_shared_local[(3)] = kernel_shared[(29)];
    kernel_shared_local[(4)] = kernel_shared[(32)];
    kernel_shared_local[(5)] = kernel_shared[(35)];
    kernel_shared_local[(6)] = kernel_shared[(56)];
    kernel_shared_local[(7)] = kernel_shared[(59)];
    kernel_shared_local[(8)] = kernel_shared[(62)];
    kernel_shared_local[(9)] = kernel_shared[(65)];
    kernel_shared_local[(10)] = kernel_shared[(68)];
    kernel_shared_local[(11)] = kernel_shared[(71)];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(11)]));
  }
  compute[(((((((int)blockIdx.z) * 392) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 392) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 196))] = compute_local[(1)];
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


        dim3 grid(2,1,16);

        dim3 block(7,14,1);

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
    outfile.open("../../evaluation_outcome/2080Ti-layers-eval-oracle.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_tvm, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<endl;
    return 0;
}


