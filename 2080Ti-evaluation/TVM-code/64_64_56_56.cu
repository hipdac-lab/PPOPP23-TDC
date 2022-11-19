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
#define N 64
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
  float compute_local[56];
  __shared__ float pad_temp_shared[960];
  __shared__ float kernel_shared[384];
  float pad_temp_shared_local[28];
  float kernel_shared_local[2];
  #pragma unroll
  for (int xx_c_init = 0; xx_c_init < 2; ++xx_c_init) {
    compute_local[(xx_c_init)] = 0.000000e+00f;
    compute_local[((xx_c_init + 28))] = 0.000000e+00f;
    compute_local[((xx_c_init + 2))] = 0.000000e+00f;
    compute_local[((xx_c_init + 30))] = 0.000000e+00f;
    compute_local[((xx_c_init + 4))] = 0.000000e+00f;
    compute_local[((xx_c_init + 32))] = 0.000000e+00f;
    compute_local[((xx_c_init + 6))] = 0.000000e+00f;
    compute_local[((xx_c_init + 34))] = 0.000000e+00f;
    compute_local[((xx_c_init + 8))] = 0.000000e+00f;
    compute_local[((xx_c_init + 36))] = 0.000000e+00f;
    compute_local[((xx_c_init + 10))] = 0.000000e+00f;
    compute_local[((xx_c_init + 38))] = 0.000000e+00f;
    compute_local[((xx_c_init + 12))] = 0.000000e+00f;
    compute_local[((xx_c_init + 40))] = 0.000000e+00f;
    compute_local[((xx_c_init + 14))] = 0.000000e+00f;
    compute_local[((xx_c_init + 42))] = 0.000000e+00f;
    compute_local[((xx_c_init + 16))] = 0.000000e+00f;
    compute_local[((xx_c_init + 44))] = 0.000000e+00f;
    compute_local[((xx_c_init + 18))] = 0.000000e+00f;
    compute_local[((xx_c_init + 46))] = 0.000000e+00f;
    compute_local[((xx_c_init + 20))] = 0.000000e+00f;
    compute_local[((xx_c_init + 48))] = 0.000000e+00f;
    compute_local[((xx_c_init + 22))] = 0.000000e+00f;
    compute_local[((xx_c_init + 50))] = 0.000000e+00f;
    compute_local[((xx_c_init + 24))] = 0.000000e+00f;
    compute_local[((xx_c_init + 52))] = 0.000000e+00f;
    compute_local[((xx_c_init + 26))] = 0.000000e+00f;
    compute_local[((xx_c_init + 54))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 120; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        pad_temp_shared[((((((int)threadIdx.z) * 240) + (((int)threadIdx.y) * 120)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((1 <= (((((int)blockIdx.y) * 2) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 60) / 30)) + ry_outer)) && ((((((int)blockIdx.y) * 2) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 60) / 30)) + ry_outer) < 57)) && (1 <= ((((int)blockIdx.x) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 30)))) && (((((int)blockIdx.x) * 28) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 30)) < 57)) ? data[(((((((((((rc_outer * 50176) + (((int)threadIdx.z) * 12544)) + (((int)threadIdx.y) * 6272)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner / 60) * 3136)) + (((int)blockIdx.y) * 112)) + (((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 60) / 30) * 56)) + (ry_outer * 56)) + (((int)blockIdx.x) * 28)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner % 30)) - 57))] : 0.000000e+00f);
      }
      #pragma unroll
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 48; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
        kernel_shared[((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 48)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = kernel[((((((((((int)blockIdx.z) * 4608) + (((int)threadIdx.z) * 1152)) + (((int)threadIdx.y) * 576)) + (rc_outer * 144)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 / 3) * 9)) + (ry_outer * 3)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 % 3)))];
      }
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 16; ++rc_inner_outer) {
        #pragma unroll
        for (int rx_inner_outer = 0; rx_inner_outer < 3; ++rx_inner_outer) {
          #pragma unroll
          for (int ax3 = 0; ax3 < 2; ++ax3) {
            pad_temp_shared_local[(ax3)] = pad_temp_shared[(((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer))];
            pad_temp_shared_local[((ax3 + 2))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 2))];
            pad_temp_shared_local[((ax3 + 4))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 4))];
            pad_temp_shared_local[((ax3 + 6))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 6))];
            pad_temp_shared_local[((ax3 + 8))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 8))];
            pad_temp_shared_local[((ax3 + 10))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 10))];
            pad_temp_shared_local[((ax3 + 12))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 12))];
            pad_temp_shared_local[((ax3 + 14))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 14))];
            pad_temp_shared_local[((ax3 + 16))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 16))];
            pad_temp_shared_local[((ax3 + 18))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 18))];
            pad_temp_shared_local[((ax3 + 20))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 20))];
            pad_temp_shared_local[((ax3 + 22))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 22))];
            pad_temp_shared_local[((ax3 + 24))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 24))];
            pad_temp_shared_local[((ax3 + 26))] = pad_temp_shared[((((((rc_inner_outer * 60) + (((int)threadIdx.y) * 30)) + ax3) + rx_inner_outer) + 26))];
          }
          kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 3)) + rx_inner_outer))];
          kernel_shared_local[(1)] = kernel_shared[(((((((int)threadIdx.z) * 48) + (rc_inner_outer * 3)) + rx_inner_outer) + 192))];
          #pragma unroll
          for (int xx_c = 0; xx_c < 2; ++xx_c) {
            compute_local[(xx_c)] = (compute_local[(xx_c)] + (pad_temp_shared_local[(xx_c)] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 28))] = (compute_local[((xx_c + 28))] + (pad_temp_shared_local[(xx_c)] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 2))] = (compute_local[((xx_c + 2))] + (pad_temp_shared_local[((xx_c + 2))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 30))] = (compute_local[((xx_c + 30))] + (pad_temp_shared_local[((xx_c + 2))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 4))] = (compute_local[((xx_c + 4))] + (pad_temp_shared_local[((xx_c + 4))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 32))] = (compute_local[((xx_c + 32))] + (pad_temp_shared_local[((xx_c + 4))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 6))] = (compute_local[((xx_c + 6))] + (pad_temp_shared_local[((xx_c + 6))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 34))] = (compute_local[((xx_c + 34))] + (pad_temp_shared_local[((xx_c + 6))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 8))] = (compute_local[((xx_c + 8))] + (pad_temp_shared_local[((xx_c + 8))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 36))] = (compute_local[((xx_c + 36))] + (pad_temp_shared_local[((xx_c + 8))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 10))] = (compute_local[((xx_c + 10))] + (pad_temp_shared_local[((xx_c + 10))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 38))] = (compute_local[((xx_c + 38))] + (pad_temp_shared_local[((xx_c + 10))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 12))] = (compute_local[((xx_c + 12))] + (pad_temp_shared_local[((xx_c + 12))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 40))] = (compute_local[((xx_c + 40))] + (pad_temp_shared_local[((xx_c + 12))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 14))] = (compute_local[((xx_c + 14))] + (pad_temp_shared_local[((xx_c + 14))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 42))] = (compute_local[((xx_c + 42))] + (pad_temp_shared_local[((xx_c + 14))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 16))] = (compute_local[((xx_c + 16))] + (pad_temp_shared_local[((xx_c + 16))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 44))] = (compute_local[((xx_c + 44))] + (pad_temp_shared_local[((xx_c + 16))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 18))] = (compute_local[((xx_c + 18))] + (pad_temp_shared_local[((xx_c + 18))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 46))] = (compute_local[((xx_c + 46))] + (pad_temp_shared_local[((xx_c + 18))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 20))] = (compute_local[((xx_c + 20))] + (pad_temp_shared_local[((xx_c + 20))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 48))] = (compute_local[((xx_c + 48))] + (pad_temp_shared_local[((xx_c + 20))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 22))] = (compute_local[((xx_c + 22))] + (pad_temp_shared_local[((xx_c + 22))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 50))] = (compute_local[((xx_c + 50))] + (pad_temp_shared_local[((xx_c + 22))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 24))] = (compute_local[((xx_c + 24))] + (pad_temp_shared_local[((xx_c + 24))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 52))] = (compute_local[((xx_c + 52))] + (pad_temp_shared_local[((xx_c + 24))] * kernel_shared_local[(1)]));
            compute_local[((xx_c + 26))] = (compute_local[((xx_c + 26))] + (pad_temp_shared_local[((xx_c + 26))] * kernel_shared_local[(0)]));
            compute_local[((xx_c + 54))] = (compute_local[((xx_c + 54))] + (pad_temp_shared_local[((xx_c + 26))] * kernel_shared_local[(1)]));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int xx_inner_inner_inner = 0; xx_inner_inner_inner < 2; ++xx_inner_inner_inner) {
    compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner))] = compute_local[(xx_inner_inner_inner)];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12544))] = compute_local[((xx_inner_inner_inner + 28))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 2))] = compute_local[((xx_inner_inner_inner + 2))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12546))] = compute_local[((xx_inner_inner_inner + 30))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 4))] = compute_local[((xx_inner_inner_inner + 4))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12548))] = compute_local[((xx_inner_inner_inner + 32))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 6))] = compute_local[((xx_inner_inner_inner + 6))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12550))] = compute_local[((xx_inner_inner_inner + 34))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 8))] = compute_local[((xx_inner_inner_inner + 8))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12552))] = compute_local[((xx_inner_inner_inner + 36))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 10))] = compute_local[((xx_inner_inner_inner + 10))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12554))] = compute_local[((xx_inner_inner_inner + 38))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12))] = compute_local[((xx_inner_inner_inner + 12))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12556))] = compute_local[((xx_inner_inner_inner + 40))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 14))] = compute_local[((xx_inner_inner_inner + 14))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12558))] = compute_local[((xx_inner_inner_inner + 42))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 16))] = compute_local[((xx_inner_inner_inner + 16))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12560))] = compute_local[((xx_inner_inner_inner + 44))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 18))] = compute_local[((xx_inner_inner_inner + 18))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12562))] = compute_local[((xx_inner_inner_inner + 46))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 20))] = compute_local[((xx_inner_inner_inner + 20))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12564))] = compute_local[((xx_inner_inner_inner + 48))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 22))] = compute_local[((xx_inner_inner_inner + 22))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12566))] = compute_local[((xx_inner_inner_inner + 50))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 24))] = compute_local[((xx_inner_inner_inner + 24))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12568))] = compute_local[((xx_inner_inner_inner + 52))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 26))] = compute_local[((xx_inner_inner_inner + 26))];
    compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 56)) + (((int)blockIdx.x) * 28)) + xx_inner_inner_inner) + 12570))] = compute_local[((xx_inner_inner_inner + 54))];
  }
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

    dim3 grid(2,28,8);
    dim3 block(1,2,4);

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


