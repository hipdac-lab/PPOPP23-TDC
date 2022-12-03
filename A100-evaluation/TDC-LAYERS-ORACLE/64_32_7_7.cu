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
#define TW 1
#define TC 4
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
  float compute_local[4];
  __shared__ float pad_temp_shared[5184];
  __shared__ float kernel_shared[4608];
  float pad_temp_shared_local[4];
  float kernel_shared_local[16];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  pad_temp_shared[((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)))] = (((((9 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 81)) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) / 81) * 49)) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 1))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 2))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 3))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 4))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 5))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 6))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 7))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 8))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 9))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 9) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 9) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 9) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 9) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 10))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 10) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 10) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 10) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 10) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 11))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 11) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 11) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 11) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 11) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 12))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 12) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 12) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 12) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 12) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 13))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 13) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 13) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 13) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 13) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 14))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 14) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 14) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 14) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 14) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 15))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 15) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 15) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 15) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 15) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 16))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 16) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 16) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 16) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 16) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 17))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 17) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 17) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 17) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 17) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 18))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 18) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 18) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 18) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 18) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 19))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 19) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 19) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 19) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 19) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 20))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 20) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 20) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 20) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 20) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 21))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 21) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 21) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 21) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 21) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 22))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 22) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 22) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 22) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 22) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 23))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 23) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 23) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 23) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 23) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 24))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 24) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 24) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 24) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 24) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 25))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 25) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 25) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 25) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 25) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 26))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 26) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 26) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 26) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 26) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 27))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 27) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 27) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 27) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 27) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 28))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 28) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 28) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 28) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 28) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 29))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 29) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 29) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 29) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 29) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 30))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 30) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 30) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 30) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 30) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 31))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 31) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 31) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 31) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 31) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 32))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 32) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 32) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 32) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 32) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 33))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 33) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 33) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 33) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 33) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 34))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 34) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 34) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 34) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 34) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 35))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 35) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 35) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 35) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 35) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 36))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 36) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 36) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 36) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 36) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 37))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 37) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 37) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 37) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 37) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 38))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 38) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 38) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 38) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 38) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 39))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 39) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 39) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 39) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 39) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 40))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 40) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 40) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 40) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 40) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 41))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 41) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 41) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 41) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 41) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 42))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 42) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 42) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 42) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 42) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 43))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 43) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 43) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 43) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 43) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 44))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 44) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 44) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 44) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 44) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 8) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 45))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 45) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 45) % 81) < 72)) && (1 <= (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9))) && ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 45) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 45) % 81) / 9) * 7)) + (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 46))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 46) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 46) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 46) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 46) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 1) % 9)) - 8))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 47))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 47) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 47) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 47) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 47) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 2) % 9)) - 8))] : 0.000000e+00f);
  if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) / 81)) < 64) {
    if (((((int)threadIdx.z) * 288) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) / 9)) < 576) {
      if ((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) < 5136) {
        if (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) < 2544) {
          pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 48))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 48) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 3) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) / 81)) < 64) {
    if (((((int)threadIdx.z) * 288) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) / 9)) < 576) {
      if ((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) < 5135) {
        if (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) < 2543) {
          pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 49))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 49) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 4) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) / 81)) < 64) {
    if (((((int)threadIdx.z) * 288) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) / 9)) < 576) {
      if ((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) < 5134) {
        if (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) < 2542) {
          pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 50))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 50) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 5) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) / 81)) < 64) {
    if (((((int)threadIdx.z) * 288) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) / 9)) < 576) {
      if ((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) < 5133) {
        if (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) < 2541) {
          pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 51))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 51) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 6) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) / 81)) < 64) {
    if (((((int)threadIdx.z) * 288) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) / 9)) < 576) {
      if ((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) < 5132) {
        if (((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) < 2540) {
          pad_temp_shared[(((((((int)threadIdx.z) * 2592) + (((int)threadIdx.y) * 371)) + (((int)threadIdx.x) * 53)) + 52))] = (((((9 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) % 81)) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) % 81) < 72)) && (1 <= ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9))) && (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9) < 8)) ? data[((((((((int)threadIdx.z) * 1568) + (((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) / 81) * 49)) + ((((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 52) % 81) / 9) * 7)) + ((((((int)threadIdx.y) * 371) + (((int)threadIdx.x) * 53)) + 7) % 9)) - 8))] : 0.000000e+00f);
        }
      }
    }
  }
  kernel_shared[((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)))] = kernel[(((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 1))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 1))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 2))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 2))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 3))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 3))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 4))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 4))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 5))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 5))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 6))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 6))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 7))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 7))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 8))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 8))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 9))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 9))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 10))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 10))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 11))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 11))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 12))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 12))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 13))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 13))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 14))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 14))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 15))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 15))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 16))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 16))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 17))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 17))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 18))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 18))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 19))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 19))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 20))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 20))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 21))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 21))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 22))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 22))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 23))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 23))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 24))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 24))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 25))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 25))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 26))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 26))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 27))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 27))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 28))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 28))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 29))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 29))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 30))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 30))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 31))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 31))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 32))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 32))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 33))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 33))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 34))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 34))];
  kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 35))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 35))];
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 508) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1524) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4572) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2268) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 36))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 36))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 508) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1524) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4571) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2267) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 37))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 37))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 508) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1524) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4570) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2266) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 12) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 38))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 38))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1523) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4569) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2265) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 39))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 39))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1523) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4568) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2264) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 40))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 40))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1523) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4567) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2263) {
            if ((((((int)blockIdx.z) * 8) + (((int)threadIdx.z) * 4)) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 13) / 192)) < 32) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 41))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 41))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1522) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4566) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2262) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 42))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 42))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1522) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4565) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2261) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 43))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 43))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 14) / 3)) < 512) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1522) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4564) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2260) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 44))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 44))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 15) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 507) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1521) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4563) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2259) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 45))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 45))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 15) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 507) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1521) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4562) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2258) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 46))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 46))];
            }
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) + 15) / 192)) < 8) {
    if (((((int)threadIdx.z) * 256) + (((((int)threadIdx.y) * 110) + (((int)threadIdx.x) * 16)) / 3)) < 507) {
      if ((((((int)threadIdx.z) * 768) + (((int)threadIdx.y) * 110)) + (((int)threadIdx.x) * 16)) < 1521) {
        if ((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) < 4561) {
          if (((((int)threadIdx.y) * 330) + (((int)threadIdx.x) * 48)) < 2257) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[(((((((int)threadIdx.z) * 2304) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 47))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 330)) + (((int)threadIdx.x) * 48)) + 47))];
            }
          }
        }
      }
    }
  }
  __syncthreads();
  for (int rc_inner_outer = 0; rc_inner_outer < 16; ++rc_inner_outer) {
    pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 81))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 162))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 243))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1152))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2304))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3456))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 9))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1161))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2313))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3465))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 18))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1170))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2322))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3474))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 27))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1179))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2331))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3483))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 82))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 163))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 244))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1153))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2305))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3457))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 10))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1162))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2314))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3466))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 19))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1171))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2323))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3475))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 28))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1180))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2332))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3484))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 83))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 164))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 245))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1154))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2306))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3458))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 11))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1163))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2315))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3467))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 20))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1172))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2324))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3476))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 29))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1181))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2333))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3485))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 90))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 171))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 252))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1155))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2307))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3459))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 12))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1164))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2316))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3468))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 21))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1173))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2325))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3477))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 30))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1182))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2334))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3486))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 91))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 172))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 253))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 4))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1156))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2308))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3460))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 13))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1165))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2317))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3469))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 22))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1174))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2326))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3478))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 31))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1183))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2335))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3487))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 92))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 173))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 254))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 5))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1157))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2309))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3461))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 14))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1166))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2318))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3470))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 23))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1175))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2327))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3479))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 32))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1184))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2336))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3488))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 99))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 180))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 261))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 6))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1158))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2310))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3462))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 15))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1167))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2319))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3471))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 24))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1176))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2328))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3480))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 33))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1185))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2337))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3489))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 19))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 100))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 181))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 262))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 7))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1159))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2311))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3463))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 16))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1168))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2320))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3472))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 25))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1177))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2329))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3481))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 34))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1186))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2338))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3490))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 101))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 182))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 324) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 263))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 8))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1160))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2312))];
    kernel_shared_local[(12)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3464))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 17))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1169))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2321))];
    kernel_shared_local[(13)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3473))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 26))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1178))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2330))];
    kernel_shared_local[(14)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3482))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 35))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 1187))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 2339))];
    kernel_shared_local[(15)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 36)) + 3491))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
  }
  compute[(((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 98))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 392) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 294))] = compute_local[(3)];
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
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[0];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[1];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[2];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 1 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[8];

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


        dim3 grid(1,1,4);

                dim3 block(7,7,2);

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
                                   cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<endl;
    return 0;
}


