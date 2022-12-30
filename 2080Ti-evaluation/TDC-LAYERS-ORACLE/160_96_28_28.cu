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
#define TH 5
#define TW 4
#define TC 16
#define C 160
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
  float compute_local[2];
  __shared__ float pad_temp_shared[640];
  __shared__ float kernel_shared[480];
  float pad_temp_shared_local[16];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 1))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 2))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 3))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 636) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 76) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 4))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 635) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 75) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 5))] = ((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) & 1))) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 480) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 60) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)))] = kernel[(((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 1) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 1) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 479) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 59) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 1))] = kernel[(((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 1) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 2) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 2) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 478) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 58) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 2))] = kernel[(((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 2) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 2) % 3)))];
            }
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) + 1) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 159) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 477) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 57) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 3))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)) + 9))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 4) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 4) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 476) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 56) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 4))] = kernel[(((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 4) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)))];
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 3))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 32))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 33))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 34))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 35))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 64))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 65))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 66))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 67))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 96))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 97))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 98))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 99))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 60))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 2))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 3))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 4))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 5))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 6))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 7))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 8))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 9))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 10))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 11))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 128))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 129))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 130))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 131))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 160))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 161))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 162))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 163))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 192))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 193))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 194))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 195))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 224))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 225))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 226))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 227))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 12))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 14))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 15))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 16))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 19))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 20))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 21))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 22))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 23))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 256))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 257))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 258))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 259))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 288))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 289))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 290))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 291))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 320))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 321))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 322))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 323))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 352))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 353))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 354))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 355))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 25))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 27))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 28))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 29))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 30))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 31))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 32))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 33))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 34))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 35))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 384))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 385))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 386))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 387))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 416))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 417))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 418))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 419))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 448))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 449))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 450))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 451))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 480))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 481))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 482))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 483))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 37))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 38))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 39))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 40))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 43))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 44))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 45))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 46))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 47))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 513))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 514))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 515))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 544))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 545))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 546))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 547))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 576))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 577))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 578))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 579))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 608))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 609))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 610))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 611))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 50))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 51))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 52))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 53))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 54))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 55))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 56))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 57))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 58))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 59))];
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
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)))] = (((1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) - 1))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 1))] = (((1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) - 1))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 2))] = (((1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) - 1))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 3))] = (((1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) - 1))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 636) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 76) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 4))] = (((1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) - 1))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 635) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 75) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 5))] = (((1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) - 1))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 480) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 60) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)) + 3))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 1) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 1) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 479) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 59) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 1))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 1) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)) + 3))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 2) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 2) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 478) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 58) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 2))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 2) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 2) % 3)) + 3))];
            }
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) + 1) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 159) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 477) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 57) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 3))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)) + 12))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 4) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 4) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 476) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 56) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 4))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 4) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)) + 3))];
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 3))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 32))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 33))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 34))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 35))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 64))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 65))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 66))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 67))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 96))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 97))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 98))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 99))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 60))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 2))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 3))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 4))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 5))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 6))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 7))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 8))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 9))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 10))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 11))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 128))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 129))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 130))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 131))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 160))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 161))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 162))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 163))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 192))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 193))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 194))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 195))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 224))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 225))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 226))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 227))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 12))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 14))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 15))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 16))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 19))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 20))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 21))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 22))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 23))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 256))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 257))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 258))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 259))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 288))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 289))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 290))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 291))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 320))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 321))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 322))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 323))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 352))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 353))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 354))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 355))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 25))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 27))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 28))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 29))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 30))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 31))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 32))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 33))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 34))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 35))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 384))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 385))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 386))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 387))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 416))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 417))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 418))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 419))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 448))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 449))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 450))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 451))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 480))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 481))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 482))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 483))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 37))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 38))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 39))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 40))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 43))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 44))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 45))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 46))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 47))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 513))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 514))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 515))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 544))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 545))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 546))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 547))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 576))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 577))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 578))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 579))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 608))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 609))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 610))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 611))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 50))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 51))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 52))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 53))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 54))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 55))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 56))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 57))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 58))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 59))];
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
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) & 15)) + 27))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 1))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 1) & 15)) + 27))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 2))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 2) & 15)) + 27))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 3))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 3) & 15)) + 27))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 636) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 76) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 4))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 4) & 15)) + 27))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) < 40) {
      if ((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) < 635) {
        if (((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) < 75) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[(((((((int)threadIdx.z) * 80) + (((int)threadIdx.y) * 40)) + (((int)threadIdx.x) * 6)) + 5))] = ((((((((int)blockIdx.y) * 2) + (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) & 1)) < 27) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) < 29)) ? data[((((((((rc_outer * 15680) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) >> 1) * 784)) + (((int)blockIdx.y) * 56)) + ((((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) >> 4)) & 1) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.y) * 40) + (((int)threadIdx.x) * 6)) + 5) & 15)) + 27))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 480) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 60) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)) + 6))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 1) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 1) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 479) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 59) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 1))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 1) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)) + 6))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 2) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 2) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 478) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 58) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 2))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 2) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 2) % 3)) + 6))];
            }
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 10) + ((((int)threadIdx.x) * 5) / 3)) + 1) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + ((((int)threadIdx.x) * 5) / 3)) < 159) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 477) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 57) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 3))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + (((((int)threadIdx.x) * 5) / 3) * 9)) + ((((int)threadIdx.x) * 5) % 3)) + 15))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 10) + (((((int)threadIdx.x) * 5) + 4) / 3)) / 20) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.y) * 10)) + (((((int)threadIdx.x) * 5) + 4) / 3)) < 160) {
        if ((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) < 476) {
          if (((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 5)) < 56) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 30)) + (((int)threadIdx.x) * 5)) + 4))] = kernel[((((((((((int)blockIdx.z) * 11520) + (((int)threadIdx.z) * 1440)) + (rc_outer * 180)) + (((int)threadIdx.y) * 90)) + ((((((int)threadIdx.x) * 5) + 4) / 3) * 9)) + (((((int)threadIdx.x) * 5) + 1) % 3)) + 6))];
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 3))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 32))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 33))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 34))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 35))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 64))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 65))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 66))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 67))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 96))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 97))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 98))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 99))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 60))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 2))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 3))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 4))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 5))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 6))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 7))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 8))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 9))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 10))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 11))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 128))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 129))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 130))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 131))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 160))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 161))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 162))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 163))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 192))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 193))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 194))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 195))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 224))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 225))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 226))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 227))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 12))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 14))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 15))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 16))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 19))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 20))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 21))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 22))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 23))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 256))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 257))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 258))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 259))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 288))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 289))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 290))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 291))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 320))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 321))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 322))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 323))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 352))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 353))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 354))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 355))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 25))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 27))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 28))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 29))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 30))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 31))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 32))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 33))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 34))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 35))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 384))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 385))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 386))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 387))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 416))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 417))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 418))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 419))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 448))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 449))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 450))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 451))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 480))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 481))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 482))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 483))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 37))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 38))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 39))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 40))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 43))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 44))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 45))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 46))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 47))];
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
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 513))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 514))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 515))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 544))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 545))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 546))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 547))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 576))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 577))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 578))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 579))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 608))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 609))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 610))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 2)) + 611))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 60) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 60) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 60) + 50))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 60) + 51))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 60) + 52))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 60) + 53))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 60) + 54))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 60) + 55))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 60) + 56))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 60) + 57))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 60) + 58))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 60) + 59))];
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
  }
  compute[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
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
		case 4: 
 		switch(write_w){
			case 1:
 			#pragma unroll
			for (unsigned int th = 0; th < 4; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 1; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 2:
 			#pragma unroll
			for (unsigned int th = 0; th < 4; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 2; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 3:
 			#pragma unroll
			for (unsigned int th = 0; th < 4; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 3; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 4:
 			#pragma unroll
			for (unsigned int th = 0; th < 4; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 4; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
		} 
		break;
		case 5: 
 		switch(write_w){
			case 1:
 			#pragma unroll
			for (unsigned int th = 0; th < 5; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 1; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 2:
 			#pragma unroll
			for (unsigned int th = 0; th < 5; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 2; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 3:
 			#pragma unroll
			for (unsigned int th = 0; th < 5; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 3; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 4:
 			#pragma unroll
			for (unsigned int th = 0; th < 5; ++th) { 
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
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[0];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[0];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[0];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[0];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[1];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[8] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[0];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[1];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[2];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[9] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[10] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[11] += shared_input[c*(TH+2)*(WPAD) + 4 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[12] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[13] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[14] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[15] += shared_input[c*(TH+2)*(WPAD) + 5 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[16] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[17] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[18] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[19] += shared_input[c*(TH+2)*(WPAD) + 6 * WPAD + tw_id * TW + 5]*data_array[8];

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


        dim3 grid(2,14,12);

        dim3 block(7,2,8);

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


