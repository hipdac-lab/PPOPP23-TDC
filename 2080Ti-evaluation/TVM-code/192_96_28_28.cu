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

#define C 192
#define N 96
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
  float compute_local[8];
  __shared__ float pad_temp_shared[4032];
  __shared__ float kernel_shared[1728];
  float pad_temp_shared_local[16];
  float kernel_shared_local[3];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)))] = (((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) && (1 <= ((int)blockIdx.x))) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 1))] = ((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 28))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 2))] = ((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 27))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 3))] = ((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 26))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 4))] = ((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 25))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 5))] = (((1 <= (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer)) && (((int)blockIdx.x) < 6)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 24))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 6))] = ((((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) && (1 <= ((int)blockIdx.x))) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) - 1))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 7))] = (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) ? data[(((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 8))] = (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) + 1))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 9))] = (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) + 2))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 10))] = (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) + 3))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 168) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 11))] = ((((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ry_outer) < 28) && (((int)blockIdx.x) < 6)) ? data[((((((((rc_outer * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 112)) + (((int)threadIdx.x) * 56)) + (ry_outer * 28)) + (((int)blockIdx.x) * 4)) + 4))] : 0.000000e+00f);
      if (((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 11) / 3)) < 576) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1728) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 72) {
              if ((((((int)blockIdx.z) * 24) + (((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) / 24)) + ((int)threadIdx.z)) < 96) {
                kernel_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)))] = kernel[((((((((((int)blockIdx.z) * 41472) + ((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) / 24) * 1728)) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + ((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) % 24) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.y) * 11) % 3)))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 1) / 3)) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 11) + 1) / 3)) < 576) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1727) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 71) {
              if ((((((int)blockIdx.z) * 24) + (((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 1) / 3)) / 24)) + ((int)threadIdx.z)) < 96) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) + 1))] = kernel[((((((((((int)blockIdx.z) * 41472) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 1) / 3)) / 24) * 1728)) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 1) / 3)) % 24) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.y) * 11) + 1) % 3)))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 2) / 3)) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 11) + 2) / 3)) < 576) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1726) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 70) {
              if ((((((int)blockIdx.z) * 24) + (((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 2) / 3)) / 24)) + ((int)threadIdx.z)) < 96) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) + 2))] = kernel[((((((((((int)blockIdx.z) * 41472) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 2) / 3)) / 24) * 1728)) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 2) / 3)) % 24) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.y) * 11) + 2) % 3)))];
              }
            }
          }
        }
      }
      if ((((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) + 1) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 11) / 3)) < 575) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1725) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 69) {
              if ((((((int)blockIdx.z) * 24) + ((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) + 1) / 24)) + ((int)threadIdx.z)) < 96) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) + 3))] = kernel[((((((((((int)blockIdx.z) * 41472) + (((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) + 1) / 24) * 1728)) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + (((((((int)threadIdx.x) * 2) + ((((int)threadIdx.y) * 11) / 3)) + 1) % 24) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.y) * 11) % 3)))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 4) / 3)) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 11) + 4) / 3)) < 576) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1724) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 68) {
              if ((((((int)blockIdx.z) * 24) + (((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 4) / 3)) / 24)) + ((int)threadIdx.z)) < 96) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) + 4))] = kernel[((((((((((int)blockIdx.z) * 41472) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 4) / 3)) / 24) * 1728)) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + ((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 4) / 3)) % 24) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.y) * 11) + 1) % 3)))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + (((((int)threadIdx.y) * 11) + 5) / 3)) / 24) + ((int)threadIdx.z)) < 24) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 11) + 5) / 3)) < 576) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) < 1723) {
            if (((((int)threadIdx.y) * 11) + (((int)threadIdx.x) * 6)) < 67) {
              if (((int)threadIdx.x) < 1) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 11)) + (((int)threadIdx.x) * 6)) + 5))] = kernel[((((((((((int)blockIdx.z) * 41472) + (((int)threadIdx.z) * 1728)) + (rc_outer * 216)) + (((int)threadIdx.x) * 18)) + ((((((int)threadIdx.y) * 11) + 5) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.y) * 11) + 2) % 3)))];
              }
            }
          }
        }
      }
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 24; ++rc_inner_outer) {
        pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 42))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 84))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 126))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 1))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 43))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 85))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 127))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 2))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 44))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 86))];
        pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 128))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 3))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 45))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 87))];
        pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 168) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + 129))];
        kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + (rc_inner_outer * 3)))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 72) + (rc_inner_outer * 3)) + 1))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 72) + (rc_inner_outer * 3)) + 2))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(0)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(0)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(0)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(1)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(2)]));
      }
    }
  }
  compute[((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 196))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 392))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 588))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 197))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 393))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 18816) + (((int)threadIdx.z) * 784)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 589))] = compute_local[(7)];
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

    dim3 grid(7,1,4);
    dim3 block(2,7,24);

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


