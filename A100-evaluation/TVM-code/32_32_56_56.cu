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

#define C 32
#define N 32
#define H 56
#define W 56

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
  __shared__ float pad_temp_shared[240];
  __shared__ float kernel_shared[576];
  float pad_temp_shared_local[12];
  float kernel_shared_local[24];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) < 24) {
      if ((((((int)threadIdx.z) * 30) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)) < 240) {
        if (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) < 30) {
          pad_temp_shared[((((((int)threadIdx.z) * 30) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) % 6))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) % 6)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) < 57)) ? data[((((((((rc_outer * 12544) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) / 6) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) / 10)) % 6) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) % 10)) - 57))] : 0.000000e+00f);
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 16) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 576) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 72) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 288)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 16) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 575) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 71) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 1))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 288)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2)) < 16) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 574) {
            if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 70) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 2))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) * 288)) + (rc_outer * 36)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) & 3) * 9)) + ((((int)threadIdx.x) % 3) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 10) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 60))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 61))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 62))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 120))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 121))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 122))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 180))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 181))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 182))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 2))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 10))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 11))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 72) + 18))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 72) + 19))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 72) + 20))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 72) + 27))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 72) + 28))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 72) + 29))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 72) + 36))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 72) + 37))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 72) + 38))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 72) + 45))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 72) + 46))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 72) + 47))];
    kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 72) + 54))];
    kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 72) + 55))];
    kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 72) + 56))];
    kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 72) + 63))];
    kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 72) + 64))];
    kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 72) + 65))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(18)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(19)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(20)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 12))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 70))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 71))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 72))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 130))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 131))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 132))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 190))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 191))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 192))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 3))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 5))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 12))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 14))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 72) + 21))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 72) + 22))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 72) + 23))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 72) + 30))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 72) + 31))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 72) + 32))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 72) + 39))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 72) + 40))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 72) + 41))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 72) + 48))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 72) + 49))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 72) + 50))];
    kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 72) + 57))];
    kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 72) + 58))];
    kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 72) + 59))];
    kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 72) + 66))];
    kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 72) + 67))];
    kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 72) + 68))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(18)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(19)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(20)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 80))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 81))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 82))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 140))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 141))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 142))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 200))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 201))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 10) + ((int)threadIdx.x)) + 202))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 6))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 7))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 15))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 16))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 17))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 72) + 24))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 72) + 25))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 72) + 26))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 72) + 33))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 72) + 34))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 72) + 35))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 72) + 42))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 72) + 43))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 72) + 44))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 72) + 51))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 72) + 52))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 72) + 53))];
    kernel_shared_local[(18)] = kernel_shared[(((((int)threadIdx.z) * 72) + 60))];
    kernel_shared_local[(19)] = kernel_shared[(((((int)threadIdx.z) * 72) + 61))];
    kernel_shared_local[(20)] = kernel_shared[(((((int)threadIdx.z) * 72) + 62))];
    kernel_shared_local[(21)] = kernel_shared[(((((int)threadIdx.z) * 72) + 69))];
    kernel_shared_local[(22)] = kernel_shared[(((((int)threadIdx.z) * 72) + 70))];
    kernel_shared_local[(23)] = kernel_shared[(((((int)threadIdx.z) * 72) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(17)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(18)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(19)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(20)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(21)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(22)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(23)]));
  }
  compute[(((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 3136))] = compute_local[(1)];
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

    dim3 grid(7,14,2);
    dim3 block(8,4,8);

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


