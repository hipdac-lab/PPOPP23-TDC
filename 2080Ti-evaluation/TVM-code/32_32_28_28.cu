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
  float compute_local[4];
  __shared__ float pad_temp_shared[864];
  __shared__ float kernel_shared[2304];
  float pad_temp_shared_local[16];
  float kernel_shared_local[4];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)))] = (((((1 <= ((((int)blockIdx.y) * 4) + ((((int)threadIdx.x) * 8) / 9))) && (((((int)blockIdx.y) * 4) + ((((int)threadIdx.x) * 8) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + ((((int)threadIdx.x) * 8) % 9)))) && (((((int)blockIdx.x) * 7) + ((((int)threadIdx.x) * 8) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((((int)threadIdx.x) * 8) / 9) * 28)) + (((int)blockIdx.x) * 7)) + ((((int)threadIdx.x) * 8) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 1) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 1) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 1) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 1) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 1) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 1) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 2) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 2) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 2) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 2) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 2) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 2) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 3) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 3) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 3) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 3) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 3) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 3) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 4) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 4) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 4) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 4) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 4) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 4) % 9)) - 29))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 5) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 5) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 5) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 5) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 5) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 5) % 9)) - 29))] : 0.000000e+00f);
    if (((((((int)threadIdx.x) * 8) + 6) / 54) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 6) + (((((int)threadIdx.x) * 8) + 6) / 9)) < 96) {
        if (((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) < 858) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 6) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 6) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 6) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 6) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 6) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 6) % 9)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 8) + 7) / 54) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 6) + (((((int)threadIdx.x) * 8) + 7) / 9)) < 96) {
        if (((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) < 857) {
          if (((int)threadIdx.x) < 6) {
            pad_temp_shared[((((((int)threadIdx.z) * 54) + (((int)threadIdx.x) * 8)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 7) / 9))) && (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 8) + 7) / 9)) < 29)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 7) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 8) + 7) % 9)) < 29)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 8) + 7) / 9) * 28)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 8) + 7) % 9)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    kernel_shared[(((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)))] = kernel[(((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 1))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 2))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 3))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 3))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 4))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 4))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 5))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 5))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 6))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 6))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 7))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 7))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 8))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 8))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 9))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 9))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 10))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 10))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 11))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 11))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 12))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 12))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 13))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 13))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 14))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 14))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 15))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 15))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 16))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 16))];
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 17))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 17))];
    if (((((((int)threadIdx.x) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 16) + ((((int)threadIdx.x) * 7) / 3)) < 254) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) < 2286) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 18))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 18))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 16) + ((((int)threadIdx.x) * 7) / 3)) < 254) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) < 2285) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 19))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 19))];
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.x) * 7) + 6) / 48) + ((int)threadIdx.z)) < 16) {
      if (((((int)threadIdx.z) * 16) + ((((int)threadIdx.x) * 7) / 3)) < 254) {
        if (((((int)threadIdx.z) * 48) + (((int)threadIdx.x) * 7)) < 762) {
          if (((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) < 2284) {
            if (((int)threadIdx.x) < 6) {
              kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.x) * 21)) + 20))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 288)) + (rc_outer * 144)) + (((int)threadIdx.x) * 21)) + 20))];
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((int)threadIdx.x))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 9))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 18))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 27))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 54))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 63))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 72))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 81))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 108))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 117))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 126))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 135))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 162))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 171))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 180))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 189))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 144))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 9))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 18))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 27))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 1))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 10))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 19))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 28))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 55))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 64))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 73))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 82))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 109))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 118))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 127))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 136))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 163))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 172))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 181))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 190))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 10))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 19))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 28))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 11))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 20))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 29))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 56))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 65))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 74))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 83))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 110))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 119))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 128))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 137))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 164))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 173))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 182))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 191))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 11))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 20))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 29))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 9))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 18))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 27))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 36))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 63))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 72))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 81))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 90))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 117))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 126))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 135))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 144))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 171))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 180))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 189))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 198))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 3))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 12))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 21))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 30))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 10))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 19))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 28))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 37))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 64))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 73))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 82))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 91))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 118))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 127))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 136))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 145))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 172))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 181))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 190))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 199))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 4))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 22))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 31))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 11))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 20))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 29))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 38))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 65))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 74))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 83))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 92))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 119))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 128))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 137))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 146))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 173))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 182))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 191))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 200))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 5))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 14))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 23))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 32))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 18))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 27))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 36))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 45))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 72))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 81))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 90))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 99))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 126))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 135))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 144))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 153))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 180))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 189))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 198))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 207))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 6))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 15))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 24))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 33))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 19))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 28))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 37))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 46))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 73))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 82))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 91))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 100))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 127))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 136))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 145))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 154))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 181))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 190))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 199))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 208))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 7))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 16))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 25))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 34))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 20))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 29))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 38))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 47))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 74))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 83))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 92))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 101))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 128))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 137))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 146))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 155))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 182))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 191))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 200))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 209))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 8))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 17))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 35))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 216))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 225))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 234))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 243))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 270))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 279))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 288))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 297))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 324))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 333))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 342))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 351))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 378))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 387))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 396))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 405))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 45))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 54))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 63))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 217))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 226))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 235))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 244))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 271))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 280))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 289))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 298))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 325))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 334))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 343))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 352))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 379))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 388))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 397))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 406))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 37))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 46))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 55))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 64))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 218))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 227))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 236))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 245))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 272))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 281))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 290))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 299))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 326))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 335))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 344))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 353))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 380))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 389))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 398))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 407))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 38))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 47))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 56))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 65))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 225))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 234))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 243))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 252))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 279))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 288))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 297))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 306))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 333))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 342))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 351))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 360))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 387))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 396))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 405))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 414))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 39))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 48))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 57))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 66))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 226))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 235))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 244))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 253))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 280))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 289))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 298))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 307))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 334))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 343))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 352))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 361))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 388))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 397))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 406))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 415))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 40))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 58))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 67))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 227))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 236))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 245))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 254))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 281))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 290))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 299))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 308))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 335))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 344))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 353))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 362))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 389))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 398))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 407))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 416))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 41))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 50))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 59))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 68))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 234))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 243))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 252))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 261))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 288))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 297))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 306))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 315))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 342))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 351))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 360))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 369))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 396))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 405))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 414))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 423))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 42))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 51))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 60))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 69))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 235))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 244))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 253))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 262))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 289))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 298))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 307))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 316))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 343))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 352))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 361))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 370))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 397))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 406))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 415))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 424))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 43))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 52))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 61))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 70))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 236))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 245))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 254))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 263))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 290))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 299))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 308))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 317))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 344))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 353))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 362))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 371))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 398))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 407))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 416))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 425))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 44))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 53))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 62))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 432))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 441))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 450))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 459))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 486))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 495))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 504))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 513))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 540))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 549))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 558))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 567))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 594))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 603))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 612))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 621))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 81))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 90))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 99))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 433))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 442))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 451))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 460))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 487))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 496))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 505))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 514))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 541))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 550))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 559))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 568))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 595))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 604))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 613))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 622))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 73))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 82))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 91))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 100))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 434))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 443))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 452))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 461))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 488))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 497))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 506))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 515))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 542))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 551))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 560))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 569))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 596))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 605))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 614))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 623))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 74))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 83))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 92))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 441))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 450))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 459))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 468))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 495))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 504))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 513))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 522))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 549))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 558))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 567))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 576))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 603))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 612))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 621))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 630))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 75))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 84))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 93))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 102))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 442))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 451))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 460))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 469))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 496))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 505))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 514))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 523))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 550))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 559))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 568))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 577))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 604))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 613))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 622))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 631))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 76))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 85))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 94))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 103))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 443))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 452))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 461))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 470))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 497))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 506))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 515))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 524))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 551))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 560))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 569))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 578))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 605))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 614))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 623))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 632))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 77))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 86))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 95))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 104))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 450))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 459))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 468))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 477))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 504))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 513))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 522))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 531))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 558))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 567))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 576))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 585))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 612))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 621))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 630))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 639))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 78))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 87))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 96))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 105))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 451))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 460))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 469))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 478))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 505))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 514))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 523))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 532))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 559))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 568))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 577))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 586))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 613))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 622))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 631))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 640))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 79))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 88))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 97))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 106))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 452))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 461))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 470))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 479))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 506))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 515))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 524))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 533))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 560))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 569))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 578))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 587))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 614))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 623))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 632))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 641))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 80))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 89))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 98))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 107))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 648))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 657))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 666))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 675))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 702))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 711))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 720))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 729))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 756))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 765))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 774))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 783))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 810))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 819))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 828))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 837))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 108))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 117))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 126))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 135))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 649))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 658))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 667))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 676))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 703))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 712))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 721))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 730))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 757))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 766))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 775))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 784))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 811))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 820))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 829))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 838))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 109))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 118))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 127))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 136))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 650))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 659))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 668))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 677))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 704))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 713))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 722))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 731))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 758))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 767))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 776))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 785))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 812))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 821))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 830))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 839))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 110))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 119))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 128))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 137))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 657))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 666))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 675))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 684))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 711))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 720))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 729))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 738))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 765))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 774))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 783))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 792))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 819))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 828))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 837))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 846))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 111))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 120))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 129))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 138))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 658))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 667))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 676))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 685))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 712))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 721))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 730))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 739))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 766))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 775))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 784))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 793))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 820))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 829))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 838))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 847))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 112))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 121))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 130))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 139))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 659))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 668))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 677))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 686))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 713))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 722))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 731))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 740))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 767))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 776))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 785))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 794))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 821))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 830))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 839))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 848))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 113))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 122))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 131))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 140))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 666))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 675))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 684))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 693))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 720))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 729))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 738))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 747))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 774))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 783))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 792))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 801))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 828))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 837))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 846))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 855))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 114))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 123))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 132))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 141))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 667))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 676))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 685))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 694))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 721))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 730))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 739))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 748))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 775))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 784))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 793))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 802))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 829))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 838))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 847))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 856))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 115))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 124))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 133))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 142))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) + 668))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((int)threadIdx.x) + 677))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((int)threadIdx.x) + 686))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((int)threadIdx.x) + 695))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((int)threadIdx.x) + 722))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((int)threadIdx.x) + 731))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((int)threadIdx.x) + 740))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((int)threadIdx.x) + 749))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((int)threadIdx.x) + 776))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((int)threadIdx.x) + 785))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((int)threadIdx.x) + 794))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((int)threadIdx.x) + 803))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((int)threadIdx.x) + 830))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((int)threadIdx.x) + 839))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((int)threadIdx.x) + 848))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((int)threadIdx.x) + 857))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 116))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 125))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 134))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 144) + 143))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
  }
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 28))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 56))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 112)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)) + 84))] = compute_local[(3)];
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

    dim3 grid(4,7,2);
    dim3 block(7,1,16);

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


