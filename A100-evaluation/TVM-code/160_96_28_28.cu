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

#define C 160
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
  float compute_local[3];
  __shared__ float pad_temp_shared[432];
  __shared__ float kernel_shared[432];
  float pad_temp_shared_local[3];
  float kernel_shared_local[9];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 20; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)))] = (((((1 <= ((((int)blockIdx.y) * 7) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 54) / 6))) && (((((int)blockIdx.y) * 7) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) / 54) * 784)) + (((int)blockIdx.y) * 196)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 6)) - 29))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 6)) - 29))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) / 54)) < 8) {
      if (((((int)threadIdx.z) * 36) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) / 6)) < 71) {
        if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) < 426) {
          if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) < 210) {
            pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)))) && (((((int)blockIdx.x) * 4) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 6)) - 29))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 54)) < 8) {
      if (((((int)threadIdx.z) * 36) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 6)) < 72) {
        if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) < 425) {
          if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) < 209) {
            if (((int)threadIdx.x) < 3) {
              pad_temp_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) % 54) / 6))) && (((((int)blockIdx.y) * 7) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) % 54) / 6)) < 29)) && (1 <= ((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)))) && (((((int)blockIdx.x) * 4) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)) < 29)) ? data[(((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 54) * 784)) + (((int)blockIdx.y) * 196)) + ((((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) % 54) / 6) * 28)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 6)) - 29))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    kernel_shared[((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) / 72) * 1440)) + (rc_outer * 72)) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) % 72)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 1))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 1) % 72)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 2))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 2) % 72)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 3))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 3) % 72)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 4))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 4) % 72)))];
    kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 5))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 5) % 72)))];
    if (((((int)threadIdx.z) * 3) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) / 72)) < 6) {
      if (((((int)threadIdx.z) * 24) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) / 9)) < 48) {
        if (((((int)threadIdx.z) * 72) + (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) / 3)) < 142) {
          if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) < 426) {
            if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) < 210) {
              kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 6))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 6) % 72)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 3) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 72)) < 6) {
      if (((((int)threadIdx.z) * 24) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 9)) < 48) {
        if (((((int)threadIdx.z) * 72) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 3)) < 144) {
          if ((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) < 425) {
            if (((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) < 209) {
              if (((int)threadIdx.x) < 3) {
                kernel_shared[(((((((int)threadIdx.z) * 216) + (((int)threadIdx.y) * 31)) + (((int)threadIdx.x) * 8)) + 7))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 4320)) + (((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) / 72) * 1440)) + (rc_outer * 72)) + ((((((int)threadIdx.y) * 31) + (((int)threadIdx.x) * 8)) + 7) % 72)))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 6) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 6))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 12))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 216))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 72))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 75))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 78))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 144))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 147))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 150))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 7))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 13))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 73))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 76))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 79))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 145))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 148))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 151))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 8))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 14))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 74))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 77))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 80))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 146))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 149))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 152))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 54))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 60))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 66))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 9))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 12))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 15))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 81))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 84))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 87))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 153))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 156))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 159))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 55))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 61))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 67))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 10))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 13))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 16))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 82))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 85))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 88))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 154))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 157))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 160))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 56))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 62))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 68))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 11))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 14))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 17))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 83))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 86))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 89))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 155))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 158))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 161))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 108))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 114))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 120))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 18))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 21))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 24))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 90))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 93))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 96))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 162))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 165))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 168))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 109))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 115))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 121))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 19))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 22))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 25))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 91))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 94))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 97))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 163))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 166))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 169))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 110))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 116))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 122))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 20))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 23))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 92))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 95))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 98))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 164))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 167))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 170))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 162))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 168))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 174))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 27))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 30))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 33))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 99))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 102))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 105))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 171))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 174))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 177))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 163))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 169))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 175))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 28))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 31))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 34))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 100))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 103))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 106))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 172))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 175))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 178))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 164))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 170))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 176))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 29))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 32))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 35))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 101))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 104))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 107))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 173))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 176))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 179))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 216))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 222))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 228))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 39))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 42))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 108))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 111))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 114))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 180))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 183))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 186))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 217))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 223))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 229))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 37))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 40))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 43))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 109))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 112))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 115))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 181))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 184))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 187))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 218))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 224))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 230))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 38))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 41))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 44))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 110))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 113))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 116))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 182))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 185))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 188))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 270))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 276))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 282))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 45))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 48))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 51))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 117))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 120))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 123))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 189))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 192))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 195))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 271))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 277))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 283))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 46))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 49))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 52))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 118))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 121))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 124))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 190))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 193))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 196))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 272))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 278))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 284))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 47))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 50))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 53))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 119))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 122))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 125))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 191))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 194))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 197))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 324))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 330))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 336))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 54))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 57))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 60))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 126))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 129))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 132))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 198))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 201))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 204))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 325))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 331))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 337))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 55))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 58))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 61))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 127))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 130))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 133))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 199))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 202))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 205))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 326))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 332))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 338))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 56))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 59))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 62))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 128))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 131))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 134))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 200))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 203))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 206))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 378))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 384))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 390))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 63))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 66))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 69))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 135))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 138))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 141))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 207))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 210))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 213))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 379))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 385))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 391))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 64))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 67))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 70))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 136))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 139))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 142))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 208))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 211))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 214))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 380))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 386))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) + 392))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 216) + 65))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 216) + 68))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 216) + 71))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 216) + 137))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 216) + 140))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 216) + 143))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 216) + 209))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 216) + 212))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 216) + 215))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
  }
  compute[(((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 2352)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 2352)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 784))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 4704) + (((int)threadIdx.z) * 2352)) + (((int)blockIdx.y) * 196)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x)) + 1568))] = compute_local[(2)];
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

    dim3 grid(7,4,16);
    dim3 block(4,7,2);

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


