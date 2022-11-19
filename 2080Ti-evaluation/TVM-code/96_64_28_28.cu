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
  float compute_local[16];
  __shared__ float pad_temp_shared[3072];
  __shared__ float kernel_shared[2304];
  float pad_temp_shared_local[2];
  float kernel_shared_local[8];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)))] = (((((1 <= (((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 55) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + (((((int)threadIdx.x) * 55) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + (((((int)threadIdx.x) * 55) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 55) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 55) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 1))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 1) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 1) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 1) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 1) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 1) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 2))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 2) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 2) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 2) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 2) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 2) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 3))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 3) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 3) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 3) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 3) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 3) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 4))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 4) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 4) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 4) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 4) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 4) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 5))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 5) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 5) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 5) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 5) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 5) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 6))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 6) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 6) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 6) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 6) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 6) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 7))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 7) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 7) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 7) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 7) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 7) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 8))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 8) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 8) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 8) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 8) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 8) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 9))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 9) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 9) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 9) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 9) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 9) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 10))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 10) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 10) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 10) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 10) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 10) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 11))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 11) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 11) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 11) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 11) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 11) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 12))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 12) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 12) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 12) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 12) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 12) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 13))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 13) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 13) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 13) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 13) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 13) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 14))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 14) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 14) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 14) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 14) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 14) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 15))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 15) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 15) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 15) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 15) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 15) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 16))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 16) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 16) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 16) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 16) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 55) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 17))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 17) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 17) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 17) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 17) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 1) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 18))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 18) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 18) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 18) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 18) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 2) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 19))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 19) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 19) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 19) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 19) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 3) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 20))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 20) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 20) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 20) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 20) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 4) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 21))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 21) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 21) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 21) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 21) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 5) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 22))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 22) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 22) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 22) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 22) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 6) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 23))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 23) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 23) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 23) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 23) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 7) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 24))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 24) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 24) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 24) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 24) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 8) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 25))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 25) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 25) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 25) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 25) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 9) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 26))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 26) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 26) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 26) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 26) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 10) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 27))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 27) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 27) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 27) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 27) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 11) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 28))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 28) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 28) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 28) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 28) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 12) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 29))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 29) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 29) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 29) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 29) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 13) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 30))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 30) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 30) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 30) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 30) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 14) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 31))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 31) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 31) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 31) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 31) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 15) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 32))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 32) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 32) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 32) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 32) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 55) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 33))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 33) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 33) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 33) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 33) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 1) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 34))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 34) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 34) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 34) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 34) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 2) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 35))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 35) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 35) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 35) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 35) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 3) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 36))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 36) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 36) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 36) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 36) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 4) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 37))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 37) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 37) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 37) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 37) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 5) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 38))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 38) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 38) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 38) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 38) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 6) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 39))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 39) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 39) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 7) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 39) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 39) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 7) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 40))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 40) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 40) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 8) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 40) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 40) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 8) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 41))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 41) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 41) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 9) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 41) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 41) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 9) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 42))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 42) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 42) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 10) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 42) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 42) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 10) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 43))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 43) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 43) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 11) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 43) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 43) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 11) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 44))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 44) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 44) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 12) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 44) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 44) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 12) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 45))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 45) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 45) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 13) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 45) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 45) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 13) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 46))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 46) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 46) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 14) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 46) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 46) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 14) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 47))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 47) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 47) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 15) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 47) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 47) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 15) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 48))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 48) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 48) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)))) && (((((int)blockIdx.x) * 14) + ((((int)threadIdx.x) * 55) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 48) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 48) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + ((((int)threadIdx.x) * 55) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 49))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 49) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 49) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 1) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 49) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 49) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 1) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 50))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 50) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 50) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 2) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 50) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 50) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 2) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 51))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 51) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 51) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 3) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 51) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 51) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 3) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 52))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 52) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 52) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 4) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 52) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 52) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 4) & 15)) - 29))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 53))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 53) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 53) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 5) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 53) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 53) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 5) & 15)) - 29))] : 0.000000e+00f);
      if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 6)) + (((((int)threadIdx.x) * 55) + 54) >> 6)) < 48) {
        if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 24)) + (((((int)threadIdx.x) * 55) + 54) >> 4)) < 192) {
          if ((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) < 3018) {
            if (((((int)threadIdx.y) * 384) + (((int)threadIdx.x) * 55)) < 1482) {
              if (((int)threadIdx.x) < 6) {
                pad_temp_shared[(((((((int)threadIdx.z) * 1536) + (((int)threadIdx.y) * 384)) + (((int)threadIdx.x) * 55)) + 54))] = (((((1 <= (((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 54) & 63) >> 4)) + ry_outer)) && ((((((int)blockIdx.y) * 4) + ((((((int)threadIdx.x) * 55) + 54) & 63) >> 4)) + ry_outer) < 29)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)))) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.x) * 55) + 6) & 15)) < 29)) ? data[(((((((((((rc_outer * 37632) + (((int)threadIdx.z) * 18816)) + (((int)threadIdx.y) * 4704)) + ((((((int)threadIdx.x) * 55) + 54) >> 6) * 784)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 55) + 54) & 63) >> 4) * 28)) + (ry_outer * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.x) * 55) + 6) & 15)) - 29))] : 0.000000e+00f);
              }
            }
          }
        }
      }
      kernel_shared[((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + (((((int)threadIdx.x) * 14) / 48) * 864)) + (rc_outer * 432)) + (((((int)threadIdx.x) * 14) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 1))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + (((((int)threadIdx.x) * 14) / 48) * 864)) + (rc_outer * 432)) + (((((int)threadIdx.x) * 14) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 2))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + (((((int)threadIdx.x) * 14) / 48) * 864)) + (rc_outer * 432)) + (((((int)threadIdx.x) * 14) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 3))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 1) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 1) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 4))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 1) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 1) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 5))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 1) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 1) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 6))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 2) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 2) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 7))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 2) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 2) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 8))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 2) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 2) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 9))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 3) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 3) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 10))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 3) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 3) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 11))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 3) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 3) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 12))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 4) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 4) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 13))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 4) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 4) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 14))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 4) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 4) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 15))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 5) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 5) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 16))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 5) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 5) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 17))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 5) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 5) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 18))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 6) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 6) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 19))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 6) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 6) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 20))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 6) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 6) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 21))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 7) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 7) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 22))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 7) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 7) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 23))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 7) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 7) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 24))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 8) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 8) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 25))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 8) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 8) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 26))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 8) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 8) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 27))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 9) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 9) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 28))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 9) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 9) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 29))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 9) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 9) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 30))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 10) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 10) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 31))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 10) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 10) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 32))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 10) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 10) % 48) * 9)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 33))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 11) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 11) % 48) * 9)) + (ry_outer * 3)))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 34))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 11) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 11) % 48) * 9)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 35))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 11) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 11) % 48) * 9)) + (ry_outer * 3)) + 2))];
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 12) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 756) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2268) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1116) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 36))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 12) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 12) % 48) * 9)) + (ry_outer * 3)))];
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 12) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 756) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2267) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1115) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 37))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 12) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 12) % 48) * 9)) + (ry_outer * 3)) + 1))];
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 12) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 756) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2266) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1114) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 38))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 12) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 12) % 48) * 9)) + (ry_outer * 3)) + 2))];
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 13) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 755) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2265) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1113) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 39))] = kernel[((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 13) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 13) % 48) * 9)) + (ry_outer * 3)))];
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 13) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 755) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2264) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1112) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 40))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 13) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 13) % 48) * 9)) + (ry_outer * 3)) + 1))];
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 14) + 13) / 48)) < 16) {
        if ((((((int)threadIdx.z) * 384) + (((int)threadIdx.y) * 96)) + (((int)threadIdx.x) * 14)) < 755) {
          if ((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) < 2263) {
            if (((((int)threadIdx.y) * 288) + (((int)threadIdx.x) * 42)) < 1111) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 1152) + (((int)threadIdx.y) * 288)) + (((int)threadIdx.x) * 42)) + 41))] = kernel[(((((((((((int)blockIdx.z) * 13824) + (((int)threadIdx.z) * 6912)) + (((int)threadIdx.y) * 1728)) + ((((((int)threadIdx.x) * 14) + 13) / 48) * 864)) + (rc_outer * 432)) + ((((((int)threadIdx.x) * 14) + 13) % 48) * 9)) + (ry_outer * 3)) + 2))];
              }
            }
          }
        }
      }
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 48; ++rc_inner_outer) {
        pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1))];
        kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1152))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 144))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1296))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 288))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1440))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 432))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1584))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(1)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(7)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 1))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1153))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 145))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1297))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 289))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1441))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 433))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1585))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(1)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(7)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 2))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 2)) + 3))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 2))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1154))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 146))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1298))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 290))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1442))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 434))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 576) + (rc_inner_outer * 3)) + 1586))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(1)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(5)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(5)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(7)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      }
    }
  }
  compute[(((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 6272))] = compute_local[(8)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 6273))] = compute_local[(9)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 784))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 7056))] = compute_local[(10)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 785))] = compute_local[(3)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 7057))] = compute_local[(11)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1568))] = compute_local[(4)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 7840))] = compute_local[(12)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1569))] = compute_local[(5)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 7841))] = compute_local[(13)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 2352))] = compute_local[(6)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 8624))] = compute_local[(14)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 2353))] = compute_local[(7)];
  compute[((((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.y) * 28)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 8625))] = compute_local[(15)];
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

    dim3 grid(2,7,4);
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


