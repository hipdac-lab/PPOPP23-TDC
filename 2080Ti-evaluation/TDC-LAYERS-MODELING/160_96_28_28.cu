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
  float compute_local[4];
  __shared__ float pad_temp_shared[2560];
  __shared__ float kernel_shared[2560];
  float pad_temp_shared_local[10];
  float kernel_shared_local[10];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
    for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
      __syncthreads();
      pad_temp_shared[((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 1))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 2))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 3))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 4))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 1) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 5))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 1) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 6))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 1) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 7))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 1) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 8))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 2) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 9))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 2) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 10))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 2) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 11))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 2) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 12))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 3) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 13))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 3) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 14))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 3) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 15))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 3) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 16))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 755))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 17))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 756))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 18))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 757))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 19))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 758))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 20))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 5) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 21))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 5) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 22))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 5) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 23))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 5) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 24))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 6) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 25))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 6) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 26))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 6) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 27))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 2) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 6) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 2) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 28))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 7) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 29))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 7) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 30))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 7) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 31))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 3) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 7) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 3) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 32))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 1539))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 33))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 1540))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 34))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 1541))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 35))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + ((((int)threadIdx.x) * 10) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + (((((int)threadIdx.x) * 10) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + (((((int)threadIdx.x) * 10) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) + 1542))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 36))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + rx_outer))) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 9) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 37))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 9) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 38))] = (((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 9) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 39))] = ((((1 <= (((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3))) && ((((((int)blockIdx.y) * 4) + ry_outer) + (((((int)threadIdx.x) * 10) + 1) & 3)) < 29)) && (((((int)blockIdx.x) * 4) + rx_outer) < 26)) ? data[((((((((((((int)threadIdx.z) * 15680) + (((int)threadIdx.y) * 7840)) + ((((((int)threadIdx.x) * 10) + 9) >> 2) * 784)) + (((int)blockIdx.y) * 112)) + (ry_outer * 28)) + ((((((int)threadIdx.x) * 10) + 1) & 3) * 28)) + (((int)blockIdx.x) * 4)) + rx_outer) - 26))] : 0.000000e+00f);
      kernel_shared[((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)))] = kernel[(((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 1))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 9))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 2))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 18))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 3))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 27))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 4))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 36))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 5))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 45))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 6))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 54))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 7))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 63))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 8))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 72))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 9))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 81))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 10))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 90))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 11))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 99))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 12))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 108))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 13))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 117))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 14))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 126))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 15))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 135))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 16))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 144))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 17))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 153))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 18))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 162))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 19))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 171))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 20))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 180))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 21))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 189))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 22))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 198))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 23))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 207))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 24))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 216))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 25))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 225))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 26))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 234))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 27))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 243))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 28))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 252))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 29))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 261))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 30))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 270))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 31))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 279))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 32))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 288))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 33))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 297))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 34))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 306))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 35))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 315))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 36))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 324))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 37))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 333))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 38))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 342))];
      kernel_shared[(((((((int)threadIdx.z) * 320) + (((int)threadIdx.y) * 160)) + (((int)threadIdx.x) * 40)) + 39))] = kernel[((((((((((int)blockIdx.z) * 23040) + (((int)threadIdx.z) * 2880)) + (((int)threadIdx.y) * 1440)) + (((int)threadIdx.x) * 360)) + (ry_outer * 3)) + rx_outer) + 351))];
      __syncthreads();
      pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 8))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 16))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 24))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 32))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 40))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 48))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 56))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 64))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 72))];
      kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 160))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1280))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1281))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 2))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1282))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 3))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1283))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 4))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1284))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 80))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 88))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 96))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 104))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 112))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 120))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 128))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 136))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 144))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 152))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 5))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1285))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 6))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1286))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 7))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1287))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 8))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1288))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 9))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1289))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 160))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 168))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 176))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 184))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 192))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 200))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 208))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 216))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 224))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 232))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 10))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1290))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 11))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1291))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 12))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1292))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 13))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1293))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 14))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1294))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 240))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 248))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 256))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 264))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 272))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 280))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 288))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 296))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 304))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 312))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 15))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1295))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 16))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1296))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 17))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1297))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 18))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1298))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 19))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1299))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 320))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 328))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 336))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 344))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 352))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 360))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 368))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 376))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 384))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 392))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 20))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1300))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 21))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1301))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 22))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1302))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 23))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1303))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 24))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1304))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 400))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 408))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 416))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 424))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 432))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 440))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 448))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 456))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 464))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 472))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 25))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1305))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 26))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1306))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 27))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1307))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 28))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1308))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 29))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1309))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 480))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 488))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 496))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 504))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 512))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 520))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 528))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 536))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 544))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 552))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 30))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1310))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 31))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1311))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 32))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1312))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 33))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1313))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 34))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1314))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 560))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 568))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 576))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 584))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 592))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 600))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 608))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 616))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 624))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 632))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 35))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1315))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 36))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1316))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 37))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1317))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 38))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1318))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 39))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1319))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 640))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 648))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 656))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 664))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 672))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 680))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 688))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 696))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 704))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 712))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 40))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1320))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 41))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1321))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 42))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1322))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 43))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1323))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 44))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1324))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 720))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 728))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 736))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 744))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 752))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 760))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 768))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 776))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 784))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 792))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 45))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1325))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 46))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1326))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 47))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1327))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 48))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1328))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 49))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1329))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 800))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 808))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 816))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 824))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 832))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 840))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 848))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 856))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 864))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 872))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 50))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1330))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 51))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1331))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 52))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1332))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 53))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1333))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 54))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1334))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 880))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 888))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 896))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 904))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 912))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 920))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 928))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 936))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 944))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 952))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 55))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1335))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 56))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1336))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 57))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1337))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 58))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1338))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 59))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1339))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 960))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 968))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 976))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 984))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 992))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1000))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1008))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1016))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1024))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1032))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 60))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1340))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 61))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1341))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 62))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1342))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 63))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1343))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 64))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1344))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1040))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1048))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1056))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1064))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1072))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1080))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1088))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1096))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1104))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1112))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 65))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1345))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 66))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1346))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 67))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1347))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 68))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1348))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 69))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1349))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1120))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1128))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1136))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1144))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1152))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1160))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1168))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1176))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1184))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1192))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 70))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1350))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 71))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1351))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 72))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1352))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 73))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1353))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 74))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1354))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1200))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1208))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1216))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1224))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1232))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1240))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1248))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1256))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1264))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1272))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 75))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1355))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 76))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1356))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 77))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1357))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 78))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1358))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 79))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1359))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1280))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1288))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1296))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1304))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1312))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1320))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1328))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1336))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1344))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1352))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 80))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1360))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 81))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1361))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 82))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1362))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 83))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1363))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 84))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1364))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1360))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1368))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1376))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1384))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1392))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1400))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1408))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1416))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1424))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1432))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 85))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1365))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 86))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1366))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 87))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1367))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 88))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1368))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 89))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1369))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1440))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1448))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1456))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1464))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1472))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1480))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1488))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1496))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1504))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1512))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 90))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1370))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 91))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1371))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 92))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1372))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 93))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1373))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 94))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1374))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1520))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1528))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1536))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1544))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1552))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1560))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1568))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1576))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1584))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1592))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 95))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1375))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 96))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1376))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 97))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1377))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 98))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1378))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 99))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1379))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1600))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1608))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1616))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1624))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1632))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1640))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1648))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1656))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1664))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1672))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 100))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1380))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 101))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1381))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 102))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1382))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 103))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1383))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 104))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1384))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1680))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1688))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1696))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1704))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1712))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1720))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1728))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1736))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1744))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1752))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 105))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1385))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 106))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1386))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 107))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1387))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 108))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1388))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 109))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1389))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1760))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1768))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1776))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1784))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1792))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1800))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1808))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1816))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1824))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1832))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 110))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1390))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 111))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1391))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 112))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1392))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 113))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1393))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 114))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1394))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1840))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1848))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1856))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1864))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1872))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1880))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1888))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1896))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1904))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1912))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 115))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1395))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 116))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1396))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 117))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1397))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 118))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1398))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 119))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1399))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1920))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1928))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1936))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1944))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1952))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1960))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1968))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1976))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1984))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 1992))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 120))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1400))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 121))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1401))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 122))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1402))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 123))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1403))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 124))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1404))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2000))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2008))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2016))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2024))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2032))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2040))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2048))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2056))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2064))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2072))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 125))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1405))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 126))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1406))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 127))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1407))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 128))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1408))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 129))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1409))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2080))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2088))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2096))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2104))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2112))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2120))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2128))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2136))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2144))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2152))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 130))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1410))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 131))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1411))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 132))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1412))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 133))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1413))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 134))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1414))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2160))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2168))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2176))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2184))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2192))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2200))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2208))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2216))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2224))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2232))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 135))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1415))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 136))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1416))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 137))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1417))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 138))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1418))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 139))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1419))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2240))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2248))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2256))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2264))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2272))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2280))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2288))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2296))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2304))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2312))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 140))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1420))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 141))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1421))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 142))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1422))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 143))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1423))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 144))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1424))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2320))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2328))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2336))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2344))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2352))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2360))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2368))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2376))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2384))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2392))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 145))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1425))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 146))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1426))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 147))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1427))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 148))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1428))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 149))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1429))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2400))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2408))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2416))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2424))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2432))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2440))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2448))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2456))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2464))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2472))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 150))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1430))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 151))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1431))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 152))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1432))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 153))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1433))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 154))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1434))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2480))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2488))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2496))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2504))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2512))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2520))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2528))];
      pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2536))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2544))];
      pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) + 2552))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 160) + 155))];
      kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1435))];
      kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 160) + 156))];
      kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1436))];
      kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 160) + 157))];
      kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1437))];
      kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 160) + 158))];
      kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1438))];
      kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 160) + 159))];
      kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 160) + 1439))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    }
  }
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 6272))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 56))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 6328))] = compute_local[(3)];
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
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[3];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[3];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[4];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[3];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[4];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[5];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[0] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[3];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[4];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[1] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[4];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[2] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[5];
		temp_result[3] += shared_input[c*(TH+2)*(WPAD) + 2 * WPAD + tw_id * TW + 5]*data_array[8];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 0]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[6];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 1]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[6];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[7];
		temp_result[4] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 2]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[6];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[7];
		temp_result[5] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 3]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[7];
		temp_result[6] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 4]*data_array[8];
		temp_result[7] += shared_input[c*(TH+2)*(WPAD) + 3 * WPAD + tw_id * TW + 5]*data_array[8];

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


        dim3 grid(7,7,6);

        dim3 block(4,2,8);

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
                                   cudnnGemmTime/time_tdc<<","<<time_tvm/time_tdc<<endl;
    return 0;
}


