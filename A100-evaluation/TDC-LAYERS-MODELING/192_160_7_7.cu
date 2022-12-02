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
#define TW 2
#define TC 16
#define C 192
#define N 160
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
  __shared__ float pad_temp_shared[432];
  __shared__ float kernel_shared[1440];
  float pad_temp_shared_local[72];
  float kernel_shared_local[72];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
      __syncthreads();
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 432) {
        pad_temp_shared[(((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)))] = (((((1 <= (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) % 9)) && ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) / 9) * 49)) + ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 431) {
        pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 1))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 1) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 1) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 1) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 1) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 430) {
        if (((int)threadIdx.y) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 2))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 2) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 2) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 2) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 2) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 429) {
        if (((int)threadIdx.y) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 3))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 3) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 3) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 3) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 3) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 428) {
        if (((int)threadIdx.y) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 4))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 4) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 4) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 4) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 4) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 427) {
        if (((int)threadIdx.y) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 5))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 5) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 5) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 5) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 5) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) < 426) {
        if (((int)threadIdx.y) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 6))] = (((((1 <= ((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 6) % 9)) && (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 6) % 9) < 8)) && (1 <= (((int)blockIdx.x) + rx_outer))) && ((((int)blockIdx.x) + rx_outer) < 8)) ? data[(((((((rc_outer * 2352) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 6) / 9) * 49)) + (((((((int)threadIdx.z) * 44) + (((int)threadIdx.y) * 7)) + 6) % 9) * 7)) + ((int)blockIdx.x)) + rx_outer) - 8))] : 0.000000e+00f);
        }
      }
      kernel_shared[(((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)))] = kernel[((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 1))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 3))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 2))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 6))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 3))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 9))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 4))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 12))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 5))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 15))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 6))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 18))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 7))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 21))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 8))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 24))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 9))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 27))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 10))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 30))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 11))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 33))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 12))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 36))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 13))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 39))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 14))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 42))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 15))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 45))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 16))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 48))];
      kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 17))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 51))];
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 10) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 474) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 1422) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 18))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 54))];
            }
          }
        }
      }
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 10) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 474) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 1421) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 19))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 57))];
            }
          }
        }
      }
      if (((((((int)threadIdx.y) * 7) + 6) / 48) + ((int)threadIdx.z)) < 10) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 7)) < 474) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) < 1420) {
            if (((int)threadIdx.y) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 21)) + 20))] = kernel[(((((((((int)blockIdx.z) * 17280) + (((int)threadIdx.z) * 1728)) + (rc_outer * 432)) + (((int)threadIdx.y) * 63)) + rx_outer) + 60))];
            }
          }
        }
      }
      __syncthreads();
      pad_temp_shared_local[(0)] = pad_temp_shared[(((int)threadIdx.y))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.y) + 1))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.y) + 2))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.y) + 9))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.y) + 10))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.y) + 11))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.y) + 18))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.y) + 19))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.y) + 20))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.y) + 27))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.y) + 28))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.y) + 29))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.y) + 36))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.y) + 37))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.y) + 38))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.y) + 45))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.y) + 46))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.y) + 47))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.y) + 54))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.y) + 55))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.y) + 56))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.y) + 63))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.y) + 64))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.y) + 65))];
      pad_temp_shared_local[(24)] = pad_temp_shared[((((int)threadIdx.y) + 72))];
      pad_temp_shared_local[(25)] = pad_temp_shared[((((int)threadIdx.y) + 73))];
      pad_temp_shared_local[(26)] = pad_temp_shared[((((int)threadIdx.y) + 74))];
      pad_temp_shared_local[(27)] = pad_temp_shared[((((int)threadIdx.y) + 81))];
      pad_temp_shared_local[(28)] = pad_temp_shared[((((int)threadIdx.y) + 82))];
      pad_temp_shared_local[(29)] = pad_temp_shared[((((int)threadIdx.y) + 83))];
      pad_temp_shared_local[(30)] = pad_temp_shared[((((int)threadIdx.y) + 90))];
      pad_temp_shared_local[(31)] = pad_temp_shared[((((int)threadIdx.y) + 91))];
      pad_temp_shared_local[(32)] = pad_temp_shared[((((int)threadIdx.y) + 92))];
      pad_temp_shared_local[(33)] = pad_temp_shared[((((int)threadIdx.y) + 99))];
      pad_temp_shared_local[(34)] = pad_temp_shared[((((int)threadIdx.y) + 100))];
      pad_temp_shared_local[(35)] = pad_temp_shared[((((int)threadIdx.y) + 101))];
      pad_temp_shared_local[(36)] = pad_temp_shared[((((int)threadIdx.y) + 108))];
      pad_temp_shared_local[(37)] = pad_temp_shared[((((int)threadIdx.y) + 109))];
      pad_temp_shared_local[(38)] = pad_temp_shared[((((int)threadIdx.y) + 110))];
      pad_temp_shared_local[(39)] = pad_temp_shared[((((int)threadIdx.y) + 117))];
      pad_temp_shared_local[(40)] = pad_temp_shared[((((int)threadIdx.y) + 118))];
      pad_temp_shared_local[(41)] = pad_temp_shared[((((int)threadIdx.y) + 119))];
      pad_temp_shared_local[(42)] = pad_temp_shared[((((int)threadIdx.y) + 126))];
      pad_temp_shared_local[(43)] = pad_temp_shared[((((int)threadIdx.y) + 127))];
      pad_temp_shared_local[(44)] = pad_temp_shared[((((int)threadIdx.y) + 128))];
      pad_temp_shared_local[(45)] = pad_temp_shared[((((int)threadIdx.y) + 135))];
      pad_temp_shared_local[(46)] = pad_temp_shared[((((int)threadIdx.y) + 136))];
      pad_temp_shared_local[(47)] = pad_temp_shared[((((int)threadIdx.y) + 137))];
      pad_temp_shared_local[(48)] = pad_temp_shared[((((int)threadIdx.y) + 144))];
      pad_temp_shared_local[(49)] = pad_temp_shared[((((int)threadIdx.y) + 145))];
      pad_temp_shared_local[(50)] = pad_temp_shared[((((int)threadIdx.y) + 146))];
      pad_temp_shared_local[(51)] = pad_temp_shared[((((int)threadIdx.y) + 153))];
      pad_temp_shared_local[(52)] = pad_temp_shared[((((int)threadIdx.y) + 154))];
      pad_temp_shared_local[(53)] = pad_temp_shared[((((int)threadIdx.y) + 155))];
      pad_temp_shared_local[(54)] = pad_temp_shared[((((int)threadIdx.y) + 162))];
      pad_temp_shared_local[(55)] = pad_temp_shared[((((int)threadIdx.y) + 163))];
      pad_temp_shared_local[(56)] = pad_temp_shared[((((int)threadIdx.y) + 164))];
      pad_temp_shared_local[(57)] = pad_temp_shared[((((int)threadIdx.y) + 171))];
      pad_temp_shared_local[(58)] = pad_temp_shared[((((int)threadIdx.y) + 172))];
      pad_temp_shared_local[(59)] = pad_temp_shared[((((int)threadIdx.y) + 173))];
      pad_temp_shared_local[(60)] = pad_temp_shared[((((int)threadIdx.y) + 180))];
      pad_temp_shared_local[(61)] = pad_temp_shared[((((int)threadIdx.y) + 181))];
      pad_temp_shared_local[(62)] = pad_temp_shared[((((int)threadIdx.y) + 182))];
      pad_temp_shared_local[(63)] = pad_temp_shared[((((int)threadIdx.y) + 189))];
      pad_temp_shared_local[(64)] = pad_temp_shared[((((int)threadIdx.y) + 190))];
      pad_temp_shared_local[(65)] = pad_temp_shared[((((int)threadIdx.y) + 191))];
      pad_temp_shared_local[(66)] = pad_temp_shared[((((int)threadIdx.y) + 198))];
      pad_temp_shared_local[(67)] = pad_temp_shared[((((int)threadIdx.y) + 199))];
      pad_temp_shared_local[(68)] = pad_temp_shared[((((int)threadIdx.y) + 200))];
      pad_temp_shared_local[(69)] = pad_temp_shared[((((int)threadIdx.y) + 207))];
      pad_temp_shared_local[(70)] = pad_temp_shared[((((int)threadIdx.y) + 208))];
      pad_temp_shared_local[(71)] = pad_temp_shared[((((int)threadIdx.y) + 209))];
      kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 144))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 1))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 2))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 3))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 4))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 5))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 6))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 7))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 8))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 9))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 10))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 11))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 144) + 12))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 144) + 13))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 144) + 14))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 144) + 15))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 144) + 16))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 144) + 17))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 144) + 18))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 144) + 19))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 144) + 20))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 144) + 21))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 144) + 22))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 144) + 23))];
      kernel_shared_local[(24)] = kernel_shared[(((((int)threadIdx.z) * 144) + 24))];
      kernel_shared_local[(25)] = kernel_shared[(((((int)threadIdx.z) * 144) + 25))];
      kernel_shared_local[(26)] = kernel_shared[(((((int)threadIdx.z) * 144) + 26))];
      kernel_shared_local[(27)] = kernel_shared[(((((int)threadIdx.z) * 144) + 27))];
      kernel_shared_local[(28)] = kernel_shared[(((((int)threadIdx.z) * 144) + 28))];
      kernel_shared_local[(29)] = kernel_shared[(((((int)threadIdx.z) * 144) + 29))];
      kernel_shared_local[(30)] = kernel_shared[(((((int)threadIdx.z) * 144) + 30))];
      kernel_shared_local[(31)] = kernel_shared[(((((int)threadIdx.z) * 144) + 31))];
      kernel_shared_local[(32)] = kernel_shared[(((((int)threadIdx.z) * 144) + 32))];
      kernel_shared_local[(33)] = kernel_shared[(((((int)threadIdx.z) * 144) + 33))];
      kernel_shared_local[(34)] = kernel_shared[(((((int)threadIdx.z) * 144) + 34))];
      kernel_shared_local[(35)] = kernel_shared[(((((int)threadIdx.z) * 144) + 35))];
      kernel_shared_local[(36)] = kernel_shared[(((((int)threadIdx.z) * 144) + 36))];
      kernel_shared_local[(37)] = kernel_shared[(((((int)threadIdx.z) * 144) + 37))];
      kernel_shared_local[(38)] = kernel_shared[(((((int)threadIdx.z) * 144) + 38))];
      kernel_shared_local[(39)] = kernel_shared[(((((int)threadIdx.z) * 144) + 39))];
      kernel_shared_local[(40)] = kernel_shared[(((((int)threadIdx.z) * 144) + 40))];
      kernel_shared_local[(41)] = kernel_shared[(((((int)threadIdx.z) * 144) + 41))];
      kernel_shared_local[(42)] = kernel_shared[(((((int)threadIdx.z) * 144) + 42))];
      kernel_shared_local[(43)] = kernel_shared[(((((int)threadIdx.z) * 144) + 43))];
      kernel_shared_local[(44)] = kernel_shared[(((((int)threadIdx.z) * 144) + 44))];
      kernel_shared_local[(45)] = kernel_shared[(((((int)threadIdx.z) * 144) + 45))];
      kernel_shared_local[(46)] = kernel_shared[(((((int)threadIdx.z) * 144) + 46))];
      kernel_shared_local[(47)] = kernel_shared[(((((int)threadIdx.z) * 144) + 47))];
      kernel_shared_local[(48)] = kernel_shared[(((((int)threadIdx.z) * 144) + 48))];
      kernel_shared_local[(49)] = kernel_shared[(((((int)threadIdx.z) * 144) + 49))];
      kernel_shared_local[(50)] = kernel_shared[(((((int)threadIdx.z) * 144) + 50))];
      kernel_shared_local[(51)] = kernel_shared[(((((int)threadIdx.z) * 144) + 51))];
      kernel_shared_local[(52)] = kernel_shared[(((((int)threadIdx.z) * 144) + 52))];
      kernel_shared_local[(53)] = kernel_shared[(((((int)threadIdx.z) * 144) + 53))];
      kernel_shared_local[(54)] = kernel_shared[(((((int)threadIdx.z) * 144) + 54))];
      kernel_shared_local[(55)] = kernel_shared[(((((int)threadIdx.z) * 144) + 55))];
      kernel_shared_local[(56)] = kernel_shared[(((((int)threadIdx.z) * 144) + 56))];
      kernel_shared_local[(57)] = kernel_shared[(((((int)threadIdx.z) * 144) + 57))];
      kernel_shared_local[(58)] = kernel_shared[(((((int)threadIdx.z) * 144) + 58))];
      kernel_shared_local[(59)] = kernel_shared[(((((int)threadIdx.z) * 144) + 59))];
      kernel_shared_local[(60)] = kernel_shared[(((((int)threadIdx.z) * 144) + 60))];
      kernel_shared_local[(61)] = kernel_shared[(((((int)threadIdx.z) * 144) + 61))];
      kernel_shared_local[(62)] = kernel_shared[(((((int)threadIdx.z) * 144) + 62))];
      kernel_shared_local[(63)] = kernel_shared[(((((int)threadIdx.z) * 144) + 63))];
      kernel_shared_local[(64)] = kernel_shared[(((((int)threadIdx.z) * 144) + 64))];
      kernel_shared_local[(65)] = kernel_shared[(((((int)threadIdx.z) * 144) + 65))];
      kernel_shared_local[(66)] = kernel_shared[(((((int)threadIdx.z) * 144) + 66))];
      kernel_shared_local[(67)] = kernel_shared[(((((int)threadIdx.z) * 144) + 67))];
      kernel_shared_local[(68)] = kernel_shared[(((((int)threadIdx.z) * 144) + 68))];
      kernel_shared_local[(69)] = kernel_shared[(((((int)threadIdx.z) * 144) + 69))];
      kernel_shared_local[(70)] = kernel_shared[(((((int)threadIdx.z) * 144) + 70))];
      kernel_shared_local[(71)] = kernel_shared[(((((int)threadIdx.z) * 144) + 71))];
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
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(24)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(25)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(26)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(27)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(28)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(29)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(30)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(31)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(32)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(33)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(34)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(35)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(36)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(37)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(38)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(39)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(40)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(41)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(42)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(43)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(44)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(45)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(46)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(47)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(48)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(49)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(50)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(51)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(52)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(53)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(54)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(55)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(56)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(57)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(58)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(59)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(60)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(61)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(62)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(63)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(64)] * kernel_shared_local[(64)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(65)] * kernel_shared_local[(65)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(66)] * kernel_shared_local[(66)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(67)] * kernel_shared_local[(67)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(68)] * kernel_shared_local[(68)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(69)] * kernel_shared_local[(69)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(70)] * kernel_shared_local[(70)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(71)] * kernel_shared_local[(71)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.y) + 216))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.y) + 217))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.y) + 218))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.y) + 225))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.y) + 226))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.y) + 227))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.y) + 234))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.y) + 235))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.y) + 236))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.y) + 243))];
      pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.y) + 244))];
      pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.y) + 245))];
      pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.y) + 252))];
      pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.y) + 253))];
      pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.y) + 254))];
      pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.y) + 261))];
      pad_temp_shared_local[(16)] = pad_temp_shared[((((int)threadIdx.y) + 262))];
      pad_temp_shared_local[(17)] = pad_temp_shared[((((int)threadIdx.y) + 263))];
      pad_temp_shared_local[(18)] = pad_temp_shared[((((int)threadIdx.y) + 270))];
      pad_temp_shared_local[(19)] = pad_temp_shared[((((int)threadIdx.y) + 271))];
      pad_temp_shared_local[(20)] = pad_temp_shared[((((int)threadIdx.y) + 272))];
      pad_temp_shared_local[(21)] = pad_temp_shared[((((int)threadIdx.y) + 279))];
      pad_temp_shared_local[(22)] = pad_temp_shared[((((int)threadIdx.y) + 280))];
      pad_temp_shared_local[(23)] = pad_temp_shared[((((int)threadIdx.y) + 281))];
      pad_temp_shared_local[(24)] = pad_temp_shared[((((int)threadIdx.y) + 288))];
      pad_temp_shared_local[(25)] = pad_temp_shared[((((int)threadIdx.y) + 289))];
      pad_temp_shared_local[(26)] = pad_temp_shared[((((int)threadIdx.y) + 290))];
      pad_temp_shared_local[(27)] = pad_temp_shared[((((int)threadIdx.y) + 297))];
      pad_temp_shared_local[(28)] = pad_temp_shared[((((int)threadIdx.y) + 298))];
      pad_temp_shared_local[(29)] = pad_temp_shared[((((int)threadIdx.y) + 299))];
      pad_temp_shared_local[(30)] = pad_temp_shared[((((int)threadIdx.y) + 306))];
      pad_temp_shared_local[(31)] = pad_temp_shared[((((int)threadIdx.y) + 307))];
      pad_temp_shared_local[(32)] = pad_temp_shared[((((int)threadIdx.y) + 308))];
      pad_temp_shared_local[(33)] = pad_temp_shared[((((int)threadIdx.y) + 315))];
      pad_temp_shared_local[(34)] = pad_temp_shared[((((int)threadIdx.y) + 316))];
      pad_temp_shared_local[(35)] = pad_temp_shared[((((int)threadIdx.y) + 317))];
      pad_temp_shared_local[(36)] = pad_temp_shared[((((int)threadIdx.y) + 324))];
      pad_temp_shared_local[(37)] = pad_temp_shared[((((int)threadIdx.y) + 325))];
      pad_temp_shared_local[(38)] = pad_temp_shared[((((int)threadIdx.y) + 326))];
      pad_temp_shared_local[(39)] = pad_temp_shared[((((int)threadIdx.y) + 333))];
      pad_temp_shared_local[(40)] = pad_temp_shared[((((int)threadIdx.y) + 334))];
      pad_temp_shared_local[(41)] = pad_temp_shared[((((int)threadIdx.y) + 335))];
      pad_temp_shared_local[(42)] = pad_temp_shared[((((int)threadIdx.y) + 342))];
      pad_temp_shared_local[(43)] = pad_temp_shared[((((int)threadIdx.y) + 343))];
      pad_temp_shared_local[(44)] = pad_temp_shared[((((int)threadIdx.y) + 344))];
      pad_temp_shared_local[(45)] = pad_temp_shared[((((int)threadIdx.y) + 351))];
      pad_temp_shared_local[(46)] = pad_temp_shared[((((int)threadIdx.y) + 352))];
      pad_temp_shared_local[(47)] = pad_temp_shared[((((int)threadIdx.y) + 353))];
      pad_temp_shared_local[(48)] = pad_temp_shared[((((int)threadIdx.y) + 360))];
      pad_temp_shared_local[(49)] = pad_temp_shared[((((int)threadIdx.y) + 361))];
      pad_temp_shared_local[(50)] = pad_temp_shared[((((int)threadIdx.y) + 362))];
      pad_temp_shared_local[(51)] = pad_temp_shared[((((int)threadIdx.y) + 369))];
      pad_temp_shared_local[(52)] = pad_temp_shared[((((int)threadIdx.y) + 370))];
      pad_temp_shared_local[(53)] = pad_temp_shared[((((int)threadIdx.y) + 371))];
      pad_temp_shared_local[(54)] = pad_temp_shared[((((int)threadIdx.y) + 378))];
      pad_temp_shared_local[(55)] = pad_temp_shared[((((int)threadIdx.y) + 379))];
      pad_temp_shared_local[(56)] = pad_temp_shared[((((int)threadIdx.y) + 380))];
      pad_temp_shared_local[(57)] = pad_temp_shared[((((int)threadIdx.y) + 387))];
      pad_temp_shared_local[(58)] = pad_temp_shared[((((int)threadIdx.y) + 388))];
      pad_temp_shared_local[(59)] = pad_temp_shared[((((int)threadIdx.y) + 389))];
      pad_temp_shared_local[(60)] = pad_temp_shared[((((int)threadIdx.y) + 396))];
      pad_temp_shared_local[(61)] = pad_temp_shared[((((int)threadIdx.y) + 397))];
      pad_temp_shared_local[(62)] = pad_temp_shared[((((int)threadIdx.y) + 398))];
      pad_temp_shared_local[(63)] = pad_temp_shared[((((int)threadIdx.y) + 405))];
      pad_temp_shared_local[(64)] = pad_temp_shared[((((int)threadIdx.y) + 406))];
      pad_temp_shared_local[(65)] = pad_temp_shared[((((int)threadIdx.y) + 407))];
      pad_temp_shared_local[(66)] = pad_temp_shared[((((int)threadIdx.y) + 414))];
      pad_temp_shared_local[(67)] = pad_temp_shared[((((int)threadIdx.y) + 415))];
      pad_temp_shared_local[(68)] = pad_temp_shared[((((int)threadIdx.y) + 416))];
      pad_temp_shared_local[(69)] = pad_temp_shared[((((int)threadIdx.y) + 423))];
      pad_temp_shared_local[(70)] = pad_temp_shared[((((int)threadIdx.y) + 424))];
      pad_temp_shared_local[(71)] = pad_temp_shared[((((int)threadIdx.y) + 425))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 72))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 73))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 74))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 75))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 144) + 76))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 144) + 77))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 144) + 78))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 144) + 79))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 144) + 80))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 144) + 81))];
      kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 144) + 82))];
      kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 144) + 83))];
      kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 144) + 84))];
      kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 144) + 85))];
      kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 144) + 86))];
      kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 144) + 87))];
      kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 144) + 88))];
      kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 144) + 89))];
      kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 144) + 90))];
      kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 144) + 91))];
      kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 144) + 92))];
      kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 144) + 93))];
      kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 144) + 94))];
      kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 144) + 95))];
      kernel_shared_local[(24)] = kernel_shared[(((((int)threadIdx.z) * 144) + 96))];
      kernel_shared_local[(25)] = kernel_shared[(((((int)threadIdx.z) * 144) + 97))];
      kernel_shared_local[(26)] = kernel_shared[(((((int)threadIdx.z) * 144) + 98))];
      kernel_shared_local[(27)] = kernel_shared[(((((int)threadIdx.z) * 144) + 99))];
      kernel_shared_local[(28)] = kernel_shared[(((((int)threadIdx.z) * 144) + 100))];
      kernel_shared_local[(29)] = kernel_shared[(((((int)threadIdx.z) * 144) + 101))];
      kernel_shared_local[(30)] = kernel_shared[(((((int)threadIdx.z) * 144) + 102))];
      kernel_shared_local[(31)] = kernel_shared[(((((int)threadIdx.z) * 144) + 103))];
      kernel_shared_local[(32)] = kernel_shared[(((((int)threadIdx.z) * 144) + 104))];
      kernel_shared_local[(33)] = kernel_shared[(((((int)threadIdx.z) * 144) + 105))];
      kernel_shared_local[(34)] = kernel_shared[(((((int)threadIdx.z) * 144) + 106))];
      kernel_shared_local[(35)] = kernel_shared[(((((int)threadIdx.z) * 144) + 107))];
      kernel_shared_local[(36)] = kernel_shared[(((((int)threadIdx.z) * 144) + 108))];
      kernel_shared_local[(37)] = kernel_shared[(((((int)threadIdx.z) * 144) + 109))];
      kernel_shared_local[(38)] = kernel_shared[(((((int)threadIdx.z) * 144) + 110))];
      kernel_shared_local[(39)] = kernel_shared[(((((int)threadIdx.z) * 144) + 111))];
      kernel_shared_local[(40)] = kernel_shared[(((((int)threadIdx.z) * 144) + 112))];
      kernel_shared_local[(41)] = kernel_shared[(((((int)threadIdx.z) * 144) + 113))];
      kernel_shared_local[(42)] = kernel_shared[(((((int)threadIdx.z) * 144) + 114))];
      kernel_shared_local[(43)] = kernel_shared[(((((int)threadIdx.z) * 144) + 115))];
      kernel_shared_local[(44)] = kernel_shared[(((((int)threadIdx.z) * 144) + 116))];
      kernel_shared_local[(45)] = kernel_shared[(((((int)threadIdx.z) * 144) + 117))];
      kernel_shared_local[(46)] = kernel_shared[(((((int)threadIdx.z) * 144) + 118))];
      kernel_shared_local[(47)] = kernel_shared[(((((int)threadIdx.z) * 144) + 119))];
      kernel_shared_local[(48)] = kernel_shared[(((((int)threadIdx.z) * 144) + 120))];
      kernel_shared_local[(49)] = kernel_shared[(((((int)threadIdx.z) * 144) + 121))];
      kernel_shared_local[(50)] = kernel_shared[(((((int)threadIdx.z) * 144) + 122))];
      kernel_shared_local[(51)] = kernel_shared[(((((int)threadIdx.z) * 144) + 123))];
      kernel_shared_local[(52)] = kernel_shared[(((((int)threadIdx.z) * 144) + 124))];
      kernel_shared_local[(53)] = kernel_shared[(((((int)threadIdx.z) * 144) + 125))];
      kernel_shared_local[(54)] = kernel_shared[(((((int)threadIdx.z) * 144) + 126))];
      kernel_shared_local[(55)] = kernel_shared[(((((int)threadIdx.z) * 144) + 127))];
      kernel_shared_local[(56)] = kernel_shared[(((((int)threadIdx.z) * 144) + 128))];
      kernel_shared_local[(57)] = kernel_shared[(((((int)threadIdx.z) * 144) + 129))];
      kernel_shared_local[(58)] = kernel_shared[(((((int)threadIdx.z) * 144) + 130))];
      kernel_shared_local[(59)] = kernel_shared[(((((int)threadIdx.z) * 144) + 131))];
      kernel_shared_local[(60)] = kernel_shared[(((((int)threadIdx.z) * 144) + 132))];
      kernel_shared_local[(61)] = kernel_shared[(((((int)threadIdx.z) * 144) + 133))];
      kernel_shared_local[(62)] = kernel_shared[(((((int)threadIdx.z) * 144) + 134))];
      kernel_shared_local[(63)] = kernel_shared[(((((int)threadIdx.z) * 144) + 135))];
      kernel_shared_local[(64)] = kernel_shared[(((((int)threadIdx.z) * 144) + 136))];
      kernel_shared_local[(65)] = kernel_shared[(((((int)threadIdx.z) * 144) + 137))];
      kernel_shared_local[(66)] = kernel_shared[(((((int)threadIdx.z) * 144) + 138))];
      kernel_shared_local[(67)] = kernel_shared[(((((int)threadIdx.z) * 144) + 139))];
      kernel_shared_local[(68)] = kernel_shared[(((((int)threadIdx.z) * 144) + 140))];
      kernel_shared_local[(69)] = kernel_shared[(((((int)threadIdx.z) * 144) + 141))];
      kernel_shared_local[(70)] = kernel_shared[(((((int)threadIdx.z) * 144) + 142))];
      kernel_shared_local[(71)] = kernel_shared[(((((int)threadIdx.z) * 144) + 143))];
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
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(24)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(25)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(26)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(27)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(28)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(29)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(30)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(31)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(32)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(33)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(34)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(35)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(36)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(37)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(38)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(39)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(40)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(41)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(42)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(43)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(44)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(45)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(46)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(47)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(48)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(49)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(50)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(51)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(52)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(53)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(54)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(55)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(56)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(57)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(58)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(59)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(60)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(61)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(62)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(63)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(64)] * kernel_shared_local[(64)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(65)] * kernel_shared_local[(65)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(66)] * kernel_shared_local[(66)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(67)] * kernel_shared_local[(67)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(68)] * kernel_shared_local[(68)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(69)] * kernel_shared_local[(69)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(70)] * kernel_shared_local[(70)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(71)] * kernel_shared_local[(71)]));
    }
  }
  compute[(((((((int)blockIdx.z) * 490) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)blockIdx.x)))] = compute_local[(0)];
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
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];

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


        dim3 grid(7,1,16);

                dim3 block(1,7,10);

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


