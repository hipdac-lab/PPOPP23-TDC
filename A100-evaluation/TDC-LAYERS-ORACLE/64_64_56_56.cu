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
#define TW 8
#define TC 16
#define C 64
#define N 64
#define H 56
#define W 56

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
  __shared__ float pad_temp_shared[768];
  __shared__ float kernel_shared[576];
  float pad_temp_shared_local[16];
  float kernel_shared_local[24];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 7) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 7) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 7) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 7) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 7) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 7) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 7) & 15)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 1) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 1) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 1) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 1) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 1) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 1) & 15)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 2) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 2) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 2) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 2) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 2) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 2) & 15)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 3) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 3) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 3) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 3) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 3) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 3) & 15)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 4) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 4) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 4) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 4) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 4) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 4) & 15)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 5) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 5) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 5) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 5) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 5) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 5) & 15)) - 57))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 6) >> 4)) / 6)) < 8) {
      if ((((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + (((((int)threadIdx.x) * 7) + 6) >> 4)) < 48) {
        if ((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) < 762) {
          if (((((int)threadIdx.y) * 48) + (((int)threadIdx.x) * 7)) < 186) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 48)) + (((int)threadIdx.x) * 7)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 6) >> 4)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 6) >> 4)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 7) + 6) & 15)) < 57)) ? data[(((((((((rc_outer * 25088) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 6) >> 4)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 7) + 6) >> 4)) % 6) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 7) + 6) & 15)) - 57))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 576) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 144) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) & 7) * 9)) + (((((int)threadIdx.x) * 2) % 3) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 575) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 143) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 1))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) & 7) * 9)) + (((((int)threadIdx.x) * 2) % 3) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 574) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 142) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 2))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) & 7) * 9)) + (((((int)threadIdx.x) * 2) % 3) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 573) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 141) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 3))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) & 7) * 9)) + ((((((int)threadIdx.x) * 2) + 1) % 3) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 572) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 140) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 4))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) & 7) * 9)) + ((((((int)threadIdx.x) * 2) + 1) % 3) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3)) < 8) {
      if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 571) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 139) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 5))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) * 576)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) & 7) * 9)) + ((((((int)threadIdx.x) * 2) + 1) % 3) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 2; ++rc_inner_outer) {
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 96))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 97))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 98))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 99))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 192))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 193))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 194))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 195))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 288))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 289))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 290))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 291))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 1))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 2))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 9))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 10))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 11))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 18))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 19))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 20))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 27))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 28))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 29))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 72))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 73))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 74))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 81))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 82))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 83))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 90))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 91))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 92))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 99))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 100))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 101))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 16))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 17))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 18))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 19))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 112))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 113))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 114))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 115))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 208))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 209))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 210))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 211))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 304))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 305))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 306))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 307))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 3))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 4))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 5))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 12))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 13))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 14))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 21))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 22))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 23))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 30))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 31))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 32))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 75))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 76))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 77))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 84))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 85))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 86))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 93))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 94))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 95))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 102))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 103))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 104))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(23)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 32))];
      pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 33))];
      pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 34))];
      pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 35))];
      pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 128))];
      pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 129))];
      pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 130))];
      pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 131))];
      pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 224))];
      pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 225))];
      pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 226))];
      pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 227))];
      pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 320))];
      pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 321))];
      pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 322))];
      pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 384) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 323))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 6))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 7))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 8))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 15))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 16))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 17))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 24))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 25))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 26))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 33))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 34))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 35))];
      kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 78))];
      kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 79))];
      kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 80))];
      kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 87))];
      kernel_shared_local[(16)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 88))];
      kernel_shared_local[(17)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 89))];
      kernel_shared_local[(18)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 96))];
      kernel_shared_local[(19)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 97))];
      kernel_shared_local[(20)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 98))];
      kernel_shared_local[(21)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 105))];
      kernel_shared_local[(22)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 106))];
      kernel_shared_local[(23)] = kernel_shared[((((((int)threadIdx.z) * 144) + (rc_inner_outer * 36)) + 107))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(12)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(13)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(14)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(15)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(15)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(16)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(16)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(17)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(17)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(18)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(18)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(19)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(19)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(20)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(20)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(9)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(21)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(21)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(10)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(22)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(22)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(23)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(23)]));
    }
  }
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 3136))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 3137))] = compute_local[(3)];
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
			case 5:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 5; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 6:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 6; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 7:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 7; ++tw) { 
					atomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);
				}
			}
			break;
			case 8:
 			#pragma unroll
			for (unsigned int th = 0; th < 1; ++th) { 
				#pragma unroll
				for (unsigned int tw = 0; tw < 8; ++tw) { 
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
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[0];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[1];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 4]*data_array[2];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 5]*data_array[0];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 5]*data_array[1];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 5]*data_array[2];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 6]*data_array[0];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 6]*data_array[1];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 6]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 7]*data_array[0];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 7]*data_array[1];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 7]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 8]*data_array[1];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 8]*data_array[2];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 0 * WPAD + tw_id * TW + 9]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[3];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[4];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 6]*data_array[3];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 6]*data_array[4];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 6]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 7]*data_array[3];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 7]*data_array[4];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 7]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 8]*data_array[4];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 8]*data_array[5];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 9]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[6];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[6];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[7];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 6]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 6]*data_array[7];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 6]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 7]*data_array[6];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 7]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 7]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 8]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 8]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 9]*data_array[8];

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


        dim3 grid(4,14,8);

                dim3 block(7,4,4);

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
    outfile.open("../../evaluation_outcome/A100-layers-eval-oracle.csv", std::ios_base::app);
    outfile << buffer;


    float difference = check_diff(out_tvm, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<
                                   time_tvm<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<cudnnWinogradeTimeNon/time_tdc<<","<<
                                   cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<","<<difference<<endl;
    return 0;
}


