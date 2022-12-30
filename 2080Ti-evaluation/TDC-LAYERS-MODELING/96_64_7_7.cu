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
#define C 96
#define N 64
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
  __shared__ float pad_temp_shared[2016];
  __shared__ float kernel_shared[384];
  float pad_temp_shared_local[8];
  float kernel_shared_local[8];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)))] = ((((7 <= (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) && ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63) < 56)) && (1 <= (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 63) * 49)) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) - 8))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 1))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) - 8))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 2))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) - 8))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 3))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) - 8))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 4))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) - 8))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 5))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) - 8))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2010) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 498) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 6))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) - 8))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 7)) < 287) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2009) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 497) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 7))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63) < 56)) && (1 <= (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) - 8))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2008) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 496) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 8))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) - 8))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2007) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 495) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 9))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) - 8))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2006) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 494) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 10))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63) < 56)) && (1 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 7))) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) - 8))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 96) {
            kernel_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = kernel[((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 95) {
            kernel_shared[(((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = kernel[(((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 3))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 63))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 126))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 189))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 252))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 315))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 378))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 12))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 15))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 21))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 7))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 70))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 133))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 259))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 385))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 448))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 16))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 19))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 22))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 77))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 140))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 203))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 266))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 329))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 455))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 11))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 20))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 23))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 504))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 567))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 693))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 756))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 819))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 945))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 27))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 30))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 33))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 36))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 39))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 45))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 511))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 574))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 700))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 763))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 826))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 889))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 952))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 25))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 28))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 31))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 34))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 37))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 40))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 43))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 46))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 518))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 581))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 644))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 707))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 770))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 896))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 959))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 26))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 29))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 32))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 35))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 38))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 44))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 47))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1008))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1071))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1134))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1197))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1260))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1386))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1449))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 51))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 54))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 57))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 60))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 63))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 66))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 69))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1015))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1141))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1204))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1267))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1330))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1393))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1456))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 49))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 52))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 55))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 58))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 61))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 64))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 67))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 70))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1022))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1085))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1148))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1211))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1337))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1400))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1463))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 50))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 53))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 56))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 59))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 62))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 65))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 68))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1575))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1638))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1701))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1764))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1827))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1890))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1953))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 75))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 78))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 81))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 84))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 87))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 90))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 93))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1582))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1645))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1708))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1771))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1834))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1897))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1960))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 73))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 76))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 79))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 82))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 85))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 88))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 91))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 94))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1526))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1589))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1652))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1715))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1778))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1841))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1904))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1967))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 74))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 77))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 80))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 83))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 86))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 89))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 92))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 95))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)))] = (((7 <= (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) && ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 63) * 49)) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) - 7))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 1))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) - 7))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 2))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) - 7))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 3))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) - 7))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 4))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) - 7))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 5))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) - 7))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2010) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 498) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 6))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) - 7))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 7)) < 287) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2009) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 497) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 7))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) - 7))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2008) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 496) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 8))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) - 7))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2007) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 495) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 9))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) - 7))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2006) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 494) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 10))] = (((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63) < 56)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) - 7))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 96) {
            kernel_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = kernel[(((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 1))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 95) {
            kernel_shared[(((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = kernel[(((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 4))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 63))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 126))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 189))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 252))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 315))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 378))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 12))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 15))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 21))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 7))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 70))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 133))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 259))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 385))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 448))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 16))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 19))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 22))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 77))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 140))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 203))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 266))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 329))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 455))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 11))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 20))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 23))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 504))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 567))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 693))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 756))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 819))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 945))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 27))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 30))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 33))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 36))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 39))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 45))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 511))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 574))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 700))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 763))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 826))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 889))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 952))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 25))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 28))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 31))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 34))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 37))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 40))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 43))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 46))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 518))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 581))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 644))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 707))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 770))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 896))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 959))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 26))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 29))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 32))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 35))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 38))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 44))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 47))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1008))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1071))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1134))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1197))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1260))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1386))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1449))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 51))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 54))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 57))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 60))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 63))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 66))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 69))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1015))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1141))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1204))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1267))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1330))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1393))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1456))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 49))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 52))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 55))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 58))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 61))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 64))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 67))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 70))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1022))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1085))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1148))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1211))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1337))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1400))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1463))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 50))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 53))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 56))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 59))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 62))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 65))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 68))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1575))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1638))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1701))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1764))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1827))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1890))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1953))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 75))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 78))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 81))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 84))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 87))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 90))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 93))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1582))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1645))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1708))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1771))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1834))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1897))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1960))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 73))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 76))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 79))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 82))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 85))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 88))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 91))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 94))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1526))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1589))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1652))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1715))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1778))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1841))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1904))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1967))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 74))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 77))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 80))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 83))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 86))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 89))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 92))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 95))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)))] = ((((7 <= (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) && ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63) < 56)) && ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 63) * 49)) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 63)) - 6))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 1))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 63)) - 6))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 2))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 63)) - 6))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 3))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 63)) - 6))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 4))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 4) % 63)) - 6))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 5))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 5) % 63)) - 6))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2010) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 498) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 6))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 6) % 63)) - 6))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) / 7)) < 287) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2009) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 497) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 7))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63) < 56)) && ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 7) % 63)) - 6))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2008) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 496) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 8))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 1) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 8) % 63)) - 6))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2007) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 495) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 9))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 2) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 9) % 63)) - 6))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 8) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63)) < 32) {
      if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 7)) < 288) {
        if ((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) < 2006) {
          if (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) < 494) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 504) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 11)) + 10))] = ((((7 <= ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63) < 56)) && (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 3) % 7) < 6)) ? data[((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + (((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) / 63) * 49)) + ((((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 11)) + 10) % 63)) - 6))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 384) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 96) {
            kernel_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] = kernel[(((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 2))];
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 4) {
      if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 128) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 383) {
          if (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) < 95) {
            kernel_shared[(((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = kernel[(((((((((int)blockIdx.z) * 3456) + (((int)threadIdx.z) * 864)) + (rc_outer * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 5))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 63))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 126))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 189))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 252))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 315))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 378))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 441))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 12))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 15))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 21))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 7))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 70))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 133))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 259))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 385))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 448))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 16))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 19))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 22))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 77))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 140))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 203))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 266))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 329))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 392))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 455))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 11))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 20))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 23))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 504))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 567))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 693))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 756))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 819))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 882))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 945))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 27))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 30))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 33))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 36))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 39))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 42))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 45))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 511))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 574))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 637))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 700))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 763))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 826))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 889))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 952))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 25))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 28))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 31))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 34))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 37))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 40))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 43))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 46))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 518))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 581))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 644))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 707))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 770))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 833))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 896))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 959))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 26))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 29))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 32))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 35))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 38))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 41))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 44))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 47))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1008))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1071))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1134))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1197))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1260))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1323))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1386))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1449))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 51))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 54))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 57))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 60))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 63))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 66))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 69))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1015))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1078))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1141))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1204))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1267))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1330))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1393))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1456))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 49))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 52))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 55))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 58))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 61))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 64))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 67))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 70))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1022))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1085))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1148))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1211))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1274))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1337))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1400))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1463))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 50))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 53))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 56))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 59))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 62))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 65))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 68))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1512))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1575))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1638))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1701))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1764))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1827))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1890))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1953))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 75))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 78))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 81))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 84))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 87))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 90))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 93))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1519))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1582))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1645))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1708))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1771))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1834))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1897))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1960))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 73))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 76))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 79))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 82))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 85))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 88))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 91))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 94))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1526))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1589))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1652))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1715))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1778))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1841))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1904))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1967))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 96) + 74))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 96) + 77))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 96) + 80))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 96) + 83))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 96) + 86))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 96) + 89))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 96) + 92))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 96) + 95))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
  }
  compute[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
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


        dim3 grid(1,1,16);

        dim3 block(7,7,4);

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
    outfile.open("../../evaluation_outcome/2080Ti-layers-eval-modeling.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_cudnn_host, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<cudnnWinogradeTimeNon/time_tdc<<","<<
                                   cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<","<<difference<<endl;
    return 0;
}


