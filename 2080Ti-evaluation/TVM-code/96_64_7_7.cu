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

#define C 96
#define N 64
#define H 7
#define W 7

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

    dim3 grid(1,1,16);
    dim3 block(7,7,4);

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


