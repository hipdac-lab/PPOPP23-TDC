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
  __shared__ float pad_temp_shared[384];
  __shared__ float kernel_shared[1152];
  float pad_temp_shared_local[4];
  float kernel_shared_local[3];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)))] = (((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) && (1 <= ((int)blockIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 1))] = ((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 28))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 2))] = ((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 27))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 3))] = ((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 26))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 4))] = ((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 25))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 5))] = (((1 <= (((int)blockIdx.y) + ((int)threadIdx.x))) && (((int)blockIdx.x) < 6)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 24))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 6))] = ((((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) && (1 <= ((int)blockIdx.x))) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) - 1))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 7))] = (((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) ? data[(((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 8))] = (((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) + 1))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 9))] = (((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) + 2))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 10))] = (((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) + 3))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 11))] = ((((((int)blockIdx.y) + ((int)threadIdx.x)) < 14) && (((int)blockIdx.x) < 6)) ? data[((((((((rc_outer * 12544) + (((int)threadIdx.z) * 1568)) + (((int)threadIdx.y) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 56)) + (((int)blockIdx.x) * 4)) + 4))] : 0.000000e+00f);
    kernel_shared[((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)))] = kernel[((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 1))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 1))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 2))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 2))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 3))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 3))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 4))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 4))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 5))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 5))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 6))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 6))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 7))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 7))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 8))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 8))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 9))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 9))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 10))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 10))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 11))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 11))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 12))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 12))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 13))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 13))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 14))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 14))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 15))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 15))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 16))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 16))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 17))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 17))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 18))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 18))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 19))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 19))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 20))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 20))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 21))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 21))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 22))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 22))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 23))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 23))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 24))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 24))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 25))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 25))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 26))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 26))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 27))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 27))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 28))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 28))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 29))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 29))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 30))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 30))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 31))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 31))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 32))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 32))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 33))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 33))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 34))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 34))];
    kernel_shared[(((((((int)threadIdx.z) * 144) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 35))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 576)) + (rc_outer * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 36)) + 35))];
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 3))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 144))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 6))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 7))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 8))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 9))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 3))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 5))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 12))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 13))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 14))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 15))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 6))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 7))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 8))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 24))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 25))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 26))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 27))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 9))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 10))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 11))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 30))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 31))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 32))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 33))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 12))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 14))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 36))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 37))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 38))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 39))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 15))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 16))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 17))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 48))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 49))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 50))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 51))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 18))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 19))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 20))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 54))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 55))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 56))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 57))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 21))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 22))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 23))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 60))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 61))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 62))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 63))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 24))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 25))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 26))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 72))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 73))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 74))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 75))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 27))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 28))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 29))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 78))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 79))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 80))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 81))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 30))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 31))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 32))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 84))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 85))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 86))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 87))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 33))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 34))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 35))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 96))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 97))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 98))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 99))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 37))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 38))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 102))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 103))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 104))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 105))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 39))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 40))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 41))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 108))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 109))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 110))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 111))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 42))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 43))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 44))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 120))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 121))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 122))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 123))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 45))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 46))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 47))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 126))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 127))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 128))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 129))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 48))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 50))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 132))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 133))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 134))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 135))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 51))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 52))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 53))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 144))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 145))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 146))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 54))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 55))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 56))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 150))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 151))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 152))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 153))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 57))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 58))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 59))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 156))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 157))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 158))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 159))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 60))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 61))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 62))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 168))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 169))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 170))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 171))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 63))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 64))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 65))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 174))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 175))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 176))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 177))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 66))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 67))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 68))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 180))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 181))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 182))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 183))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 69))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 70))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 192))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 193))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 194))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 195))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 73))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 74))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 198))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 199))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 200))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 201))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 75))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 76))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 77))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 204))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 205))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 206))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 207))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 78))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 79))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 80))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 216))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 217))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 218))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 219))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 81))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 82))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 83))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 222))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 223))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 224))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 225))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 84))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 85))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 86))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 228))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 229))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 230))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 231))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 87))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 88))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 89))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 240))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 241))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 242))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 243))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 90))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 91))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 92))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 246))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 247))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 248))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 249))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 93))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 94))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 95))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 252))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 253))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 254))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 255))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 97))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 98))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 264))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 265))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 266))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 267))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 99))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 100))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 270))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 271))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 272))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 273))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 102))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 103))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 104))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 276))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 277))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 278))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 279))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 105))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 106))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 107))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 288))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 289))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 290))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 291))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 108))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 109))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 110))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 294))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 295))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 296))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 297))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 111))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 112))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 113))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 300))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 301))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 302))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 303))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 114))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 115))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 116))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 312))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 313))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 314))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 315))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 117))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 118))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 119))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 318))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 319))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 320))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 321))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 120))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 121))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 122))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 324))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 325))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 326))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 327))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 123))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 124))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 125))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 336))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 337))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 338))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 339))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 126))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 127))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 128))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 342))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 343))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 344))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 345))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 129))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 130))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 131))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 348))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 349))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 350))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 351))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 132))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 133))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 134))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 360))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 361))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 362))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 363))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 135))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 136))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 137))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 366))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 367))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 368))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 369))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 138))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 139))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 140))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 372))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 373))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 374))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + (((int)threadIdx.x) * 2)) + 375))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 144) + 141))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 144) + 142))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 144) + 143))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
  }
  compute[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
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

    dim3 grid(7,14,4);
    dim3 block(2,2,8);

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


