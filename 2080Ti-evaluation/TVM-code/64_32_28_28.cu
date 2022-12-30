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

#define C 64
#define N 32
#define H 28
#define W 28

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
extern "C" __global__ void default_function_kernel0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[216];
  __shared__ float kernel_shared[144];
  float pad_temp_shared_local[3];
  float kernel_shared_local[6];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 108) + (((int)threadIdx.y) * 27)) + (((int)threadIdx.x) * 4)))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 4) / 9)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 4) / 9)) % 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + ((((int)threadIdx.x) * 4) % 9)))) && (((((int)blockIdx.x) * 7) + ((((int)threadIdx.x) * 4) % 9)) < 29)) ? data[(((((((((rc_outer * 3136) + (((int)threadIdx.z) * 1568)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 4) / 9)) / 6) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.y) * 3) + ((((int)threadIdx.x) * 4) / 9)) % 6) * 28)) + (((int)blockIdx.x) * 7)) + ((((int)threadIdx.x) * 4) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 108) + (((int)threadIdx.y) * 27)) + (((int)threadIdx.x) * 4)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 1) / 9)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 1) / 9)) % 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 1) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 1) % 9)) < 29)) ? data[(((((((((rc_outer * 3136) + (((int)threadIdx.z) * 1568)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 1) / 9)) / 6) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 1) / 9)) % 6) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 4) + 1) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 108) + (((int)threadIdx.y) * 27)) + (((int)threadIdx.x) * 4)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 2) / 9)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 2) / 9)) % 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 2) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 2) % 9)) < 29)) ? data[(((((((((rc_outer * 3136) + (((int)threadIdx.z) * 1568)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 2) / 9)) / 6) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 2) / 9)) % 6) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 4) + 2) % 9)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 3) / 9)) / 6)) < 4) {
      if ((((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + (((((int)threadIdx.x) * 4) + 3) / 9)) < 24) {
        if ((((((int)threadIdx.z) * 108) + (((int)threadIdx.y) * 27)) + (((int)threadIdx.x) * 4)) < 213) {
          if (((((int)threadIdx.y) * 27) + (((int)threadIdx.x) * 4)) < 105) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 108) + (((int)threadIdx.y) * 27)) + (((int)threadIdx.x) * 4)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 3) / 9)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 3) / 9)) % 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 3) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 4) + 3) % 9)) < 29)) ? data[(((((((((rc_outer * 3136) + (((int)threadIdx.z) * 1568)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 3) / 9)) / 6) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.y) * 3) + (((((int)threadIdx.x) * 4) + 3) / 9)) % 6) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 4) + 3) % 9)) - 29))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 4) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 16) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 48) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 144) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 72) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 4) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 16) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 48) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 143) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 71) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 1))] = kernel[((((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 4) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 16) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 48) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 142) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 70) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 2))] = kernel[((((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 9) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 2))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 36))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 73))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 2))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 74))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 11))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 3))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 75))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 4))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 76))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 5))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 77))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 19))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 20))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 78))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 7))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 79))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 8))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 80))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 54))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 55))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 56))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 9))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 81))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 82))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 11))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 83))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 63))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 64))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 65))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 12))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 84))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 13))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 85))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 86))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 72))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 73))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 74))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 15))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 87))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 16))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 88))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 17))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 89))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 108))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 109))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 110))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 18))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 90))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 19))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 91))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 20))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 92))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 117))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 118))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 119))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 21))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 93))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 22))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 94))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 23))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 95))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 126))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 127))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 128))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 24))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 25))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 97))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 26))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 98))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 162))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 163))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 164))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 27))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 99))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 28))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 100))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 29))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 171))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 172))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 173))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 30))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 102))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 31))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 103))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 32))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 104))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 180))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 181))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 182))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 33))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 105))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 34))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 106))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 35))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 107))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
  }
  compute[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 1568))] = compute_local[(1)];
}





float check_diff(float *x, float *y, unsigned int size){
    float diff = 0.0f;
    #pragma omp parallel for reduction(+ : diff)
    for(unsigned int i=0;i<size;++i){
        diff += abs(x[i] - y[i]);
    }
    return diff;
}
void pad_input(float * x, float *y){
    #pragma omp parallel for
    for(unsigned int i=0;i<(H + 2)*(W+2)*C;++i){
        y[i] = 0.0f;
    }
    #pragma omp parallel for
    for(unsigned int c=0;c<C;++c){
        for(unsigned int h=0;h<H;++h){
            for(unsigned int w=0;w<W;++w){
                unsigned int h_padded = h + 1;
                unsigned int w_padded = w + 1;
                y[c*(H+2)*(W+2) + h_padded*(W+2) + w_padded] = x[c*(H)*(W) + h*(W) + w];
            }
        }
    }
}
int main(void){
    float *input = new float[C*H*W];
    time_t t;
    srand((unsigned) time(&t));
    for(int i =0;i<C*H*W;++i){
        input[i] = rand() % 10;
    }
    float * padded_input = new float[C*(H+2)*(W+2)];
    pad_input(input, padded_input);
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

    dim3 grid(4,7,8);
    dim3 block(7,4,2);

    float * paddedInputDevice;
    chkerr(cudaMalloc(&paddedInputDevice, C * (H + 2) * (W + 2) * sizeof(float)));
    chkerr(cudaMemcpy(paddedInputDevice, padded_input, C * (H + 2) * (W + 2) * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(event_start);
    default_function_kernel0<<<grid, block>>>(device_input, device_K, device_out);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    float time_tdc;
    cudaEventElapsedTime(&time_tdc, event_start, event_stop);
    float *out_tdc = new float[N*H*W];
    cudaMemcpy(out_tdc,device_out,N*H*W*sizeof(float),cudaMemcpyDeviceToHost);

    float difference = check_diff(out_cudnn_host, out_tdc, N*H*W);
    cout<<N<<","<<C<<","<<H<<","<<W<<","<<cudnnFFTTime<<","<<cudnnWinogradeTimeNon<<","<<cudnnGemmTime<<","<<time_tdc<<","<<cudnnFFTTime/time_tdc<<","<<cudnnWinogradeTimeNon/time_tdc<<","<<cudnnGemmTime/time_tdc<<endl;
    return 0;
}


