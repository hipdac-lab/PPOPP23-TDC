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
#define TW 4
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
  float compute_local[4];
  __shared__ float pad_temp_shared[1536];
  __shared__ float kernel_shared[2304];
  float pad_temp_shared_local[48];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)))] = ((((1 <= ((((int)threadIdx.y) * 7) & 15)) && (((((int)threadIdx.y) * 7) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + (((((int)threadIdx.y) * 7) >> 4) * 196)) + (((((int)threadIdx.y) * 7) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 1))] = ((((1 <= ((((int)threadIdx.y) * 7) & 15)) && (((((int)threadIdx.y) * 7) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + (((((int)threadIdx.y) * 7) >> 4) * 196)) + (((((int)threadIdx.y) * 7) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 2))] = ((((1 <= (((((int)threadIdx.y) * 7) + 1) & 15)) && ((((((int)threadIdx.y) * 7) + 1) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 1) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 1) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 3))] = ((((1 <= (((((int)threadIdx.y) * 7) + 1) & 15)) && ((((((int)threadIdx.y) * 7) + 1) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 1) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 1) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 4))] = ((((1 <= (((((int)threadIdx.y) * 7) + 2) & 15)) && ((((((int)threadIdx.y) * 7) + 2) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 2) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 2) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 5))] = ((((1 <= (((((int)threadIdx.y) * 7) + 2) & 15)) && ((((((int)threadIdx.y) * 7) + 2) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 2) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 2) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 6))] = ((((1 <= (((((int)threadIdx.y) * 7) + 3) & 15)) && ((((((int)threadIdx.y) * 7) + 3) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 3) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 3) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 7))] = ((((1 <= (((((int)threadIdx.y) * 7) + 3) & 15)) && ((((((int)threadIdx.y) * 7) + 3) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 3) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 3) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 8))] = ((((1 <= (((((int)threadIdx.y) * 7) + 4) & 15)) && ((((((int)threadIdx.y) * 7) + 4) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 4) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 4) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 9))] = ((((1 <= (((((int)threadIdx.y) * 7) + 4) & 15)) && ((((((int)threadIdx.y) * 7) + 4) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 4) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 4) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 10))] = ((((1 <= (((((int)threadIdx.y) * 7) + 5) & 15)) && ((((((int)threadIdx.y) * 7) + 5) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 5) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 5) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 11))] = ((((1 <= (((((int)threadIdx.y) * 7) + 5) & 15)) && ((((((int)threadIdx.y) * 7) + 5) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 5) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 5) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + 6) >> 4)) < 48) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 762) {
          if (((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) < 1524) {
            if (((int)threadIdx.y) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 12))] = ((((1 <= (((((int)threadIdx.y) * 7) + 6) & 15)) && ((((((int)threadIdx.y) * 7) + 6) & 15) < 15)) && (1 <= ((((int)blockIdx.x) * 2) + rx_outer))) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 6) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 6) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 7) + 6) >> 4)) < 48) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 762) {
          if (((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) < 1523) {
            if (((int)threadIdx.y) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + 13))] = ((((1 <= (((((int)threadIdx.y) * 7) + 6) & 15)) && ((((((int)threadIdx.y) * 7) + 6) & 15) < 15)) && (((((int)blockIdx.x) * 2) + rx_outer) < 14)) ? data[((((((((rc_outer * 9408) + (((int)threadIdx.z) * 588)) + ((((((int)threadIdx.y) * 7) + 6) >> 4) * 196)) + ((((((int)threadIdx.y) * 7) + 6) & 15) * 14)) + (((int)blockIdx.x) * 2)) + rx_outer) - 14))] : 0.000000e+00f);
            }
          }
        }
      }
      kernel_shared[(((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)))] = kernel[((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 1))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 3))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 2))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 6))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 3))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 9))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 4))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 12))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 5))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 15))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 6))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 18))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 7))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 21))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 8))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 24))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 9))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 27))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 10))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 30))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 11))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 33))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 12))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 36))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 13))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 39))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 14))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 42))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 15))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 45))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 16))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 48))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 17))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 51))];
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 2286) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 18))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 54))];
            }
          }
        }
      }
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 2285) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 19))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 57))];
            }
          }
        }
      }
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 2284) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 20))] = kernel[(((((((((int)blockIdx.z) * 27648) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 60))];
            }
          }
        }
      }
      __syncthreads();
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.y) * 2))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 14))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 15))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 32))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 46))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 33))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 47))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 64))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 78))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 65))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 79))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 96))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 110))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 97))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 111))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 128))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 142))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 129))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 143))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 160))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 174))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 161))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 175))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 192))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 206))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 193))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 207))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 224))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 238))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 225))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 239))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 256))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 270))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 257))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 271))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 288))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 302))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 289))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 303))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 320))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 334))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 321))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 335))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 352))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 366))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 353))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 367))];
      kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 144))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 3))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 6))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 9))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 12))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 15))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 18))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 21))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 24))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 27))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 30))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 33))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 2))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 16))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 3))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 17))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 34))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 48))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 35))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 49))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 66))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 80))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 67))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 81))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 98))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 112))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 99))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 113))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 130))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 144))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 131))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 145))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 162))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 176))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 163))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 177))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 194))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 208))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 195))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 209))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 226))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 240))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 227))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 241))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 258))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 272))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 259))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 273))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 290))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 304))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 291))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 305))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 322))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 336))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 323))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 337))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 354))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 368))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 355))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 369))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 1))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 4))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 7))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 10))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 13))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 16))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 19))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 22))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 25))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 28))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 31))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 34))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 4))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 18))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 5))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 19))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 36))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 50))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 37))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 51))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 68))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 82))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 69))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 83))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 100))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 114))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 101))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 115))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 132))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 146))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 133))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 147))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 164))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 178))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 165))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 179))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 196))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 210))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 197))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 211))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 228))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 242))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 229))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 243))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 260))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 274))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 261))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 275))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 292))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 306))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 293))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 307))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 324))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 338))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 325))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 339))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 356))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 370))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 357))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 371))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 2))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 5))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 8))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 11))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 14))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 17))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 20))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 23))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 26))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 29))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 32))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 35))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 384))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 398))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 385))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 399))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 416))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 430))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 417))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 431))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 448))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 462))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 449))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 463))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 480))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 494))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 481))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 495))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 512))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 526))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 513))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 527))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 544))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 558))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 545))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 559))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 576))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 590))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 577))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 591))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 608))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 622))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 609))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 623))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 640))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 654))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 641))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 655))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 672))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 686))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 673))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 687))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 704))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 718))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 705))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 719))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 736))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 750))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 737))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 751))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 36))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 39))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 42))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 45))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 48))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 51))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 54))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 57))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 60))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 63))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 66))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 69))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 386))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 400))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 387))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 401))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 418))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 432))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 419))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 433))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 450))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 464))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 451))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 465))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 482))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 496))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 483))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 497))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 514))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 528))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 515))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 529))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 546))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 560))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 547))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 561))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 578))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 592))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 579))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 593))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 610))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 624))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 611))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 625))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 642))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 656))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 643))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 657))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 674))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 688))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 675))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 689))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 706))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 720))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 707))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 721))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 738))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 752))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 739))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 753))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 37))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 40))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 43))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 46))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 49))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 52))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 55))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 58))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 61))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 64))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 67))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 70))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 388))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 402))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 389))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 403))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 420))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 434))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 421))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 435))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 452))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 466))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 453))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 467))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 484))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 498))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 485))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 499))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 516))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 530))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 517))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 531))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 548))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 562))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 549))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 563))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 580))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 594))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 581))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 595))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 612))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 626))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 613))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 627))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 644))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 658))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 645))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 659))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 676))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 690))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 677))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 691))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 708))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 722))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 709))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 723))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 740))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 754))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 741))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 755))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 38))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 41))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 44))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 47))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 50))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 53))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 56))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 59))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 62))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 65))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 68))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 71))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 768))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 782))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 769))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 783))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 800))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 814))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 801))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 815))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 832))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 846))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 833))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 847))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 864))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 878))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 865))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 879))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 896))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 910))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 897))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 911))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 928))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 942))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 929))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 943))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 960))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 974))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 961))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 975))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 992))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1006))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 993))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1007))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1024))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1038))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1025))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1039))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1056))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1070))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1057))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1071))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1088))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1102))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1089))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1103))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1120))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1134))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1121))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1135))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 72))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 75))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 78))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 81))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 84))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 87))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 90))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 93))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 96))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 99))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 102))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 105))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 770))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 784))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 771))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 785))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 802))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 816))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 803))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 817))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 834))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 848))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 835))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 849))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 866))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 880))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 867))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 881))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 898))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 912))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 899))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 913))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 930))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 944))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 931))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 945))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 962))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 976))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 963))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 977))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 994))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1008))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 995))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1009))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1026))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1040))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1027))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1041))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1058))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1072))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1059))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1073))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1090))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1104))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1091))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1105))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1122))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1136))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1123))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1137))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 73))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 76))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 79))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 82))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 85))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 88))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 91))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 94))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 97))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 100))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 103))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 106))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 772))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 786))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 773))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 787))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 804))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 818))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 805))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 819))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 836))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 850))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 837))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 851))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 868))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 882))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 869))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 883))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 900))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 914))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 901))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 915))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 932))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 946))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 933))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 947))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 964))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 978))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 965))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 979))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 996))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1010))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 997))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1011))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1028))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1042))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1029))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1043))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1060))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1074))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1061))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1075))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1092))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1106))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1093))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1107))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1124))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1138))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1125))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1139))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 74))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 77))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 80))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 83))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 86))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 89))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 92))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 95))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 98))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 101))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 104))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 107))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1152))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1166))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1153))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1167))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1184))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1198))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1185))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1199))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1216))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1230))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1217))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1231))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1248))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1262))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1249))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1263))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1280))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1294))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1281))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1295))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1312))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1326))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1313))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1327))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1344))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1358))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1345))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1359))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1376))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1390))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1377))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1391))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1408))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1422))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1409))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1423))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1440))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1454))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1441))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1455))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1472))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1486))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1473))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1487))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1504))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1518))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1505))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1519))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 108))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 111))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 114))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 117))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 120))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 123))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 126))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 129))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 132))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 135))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 138))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 141))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1154))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1168))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1155))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1169))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1186))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1200))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1187))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1201))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1218))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1232))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1219))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1233))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1250))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1264))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1251))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1265))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1282))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1296))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1283))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1297))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1314))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1328))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1315))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1329))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1346))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1360))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1347))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1361))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1378))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1392))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1379))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1393))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1410))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1424))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1411))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1425))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1442))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1456))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1443))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1457))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1474))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1488))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1475))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1489))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1506))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1520))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1507))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1521))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 109))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 112))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 115))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 118))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 121))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 124))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 127))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 130))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 133))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 136))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 139))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 142))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1156))];
      pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1170))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1157))];
      pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1171))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1188))];
      pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1202))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1189))];
      pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1203))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1220))];
      pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1234))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1221))];
      pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1235))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1252))];
      pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1266))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1253))];
      pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1267))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1284))];
      pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1298))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1285))];
      pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1299))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1316))];
      pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1330))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1317))];
      pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1331))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1348))];
      pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1362))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1349))];
      pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1363))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1380))];
      pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1394))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1381))];
      pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1395))];
      pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1412))];
      pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1426))];
      pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1413))];
      pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1427))];
      pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1444))];
      pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1458))];
      pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1445))];
      pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1459))];
      pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1476))];
      pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1490))];
      pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1477))];
      pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1491))];
      pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1508))];
      pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1522))];
      pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1509))];
      pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.y) * 2) + 1523))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 110))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 113))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 116))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 119))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 122))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 125))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 128))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 131))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 134))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 137))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 140))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 143))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(1)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(2)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(4)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(6)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(7)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(9)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(10)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(11)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    }
  }
  compute[(((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 2)) + 98))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 2)) + 99))] = compute_local[(3)];
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
			case 3:
 			#pragma unroll
			for (unsigned int th = 0; th < 2; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 3; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 4:
 			#pragma unroll
			for (unsigned int th = 0; th < 2; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 4; ++tw) { 
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
			case 3:
 			#pragma unroll
			for (unsigned int th = 0; th < 3; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 3; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 4:
 			#pragma unroll
			for (unsigned int th = 0; th < 3; ++th) { 
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
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[0];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[0];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[0];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[0];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 5]*data_array[8];

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


        dim3 grid(7,1,6);

        dim3 block(1,7,16);

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


