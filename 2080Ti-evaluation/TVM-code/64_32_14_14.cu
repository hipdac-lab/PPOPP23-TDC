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
#define H 14
#define W 14

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
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[1152];
  float pad_temp_shared_local[8];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[(((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((int)threadIdx.x) * 74) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 1) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 2) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 3) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 4) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 5) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 6) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 7) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 8) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 9) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 10) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 10) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 10) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 11) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 11) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 11) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 12) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 12) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 12) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 13) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 13) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 13) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 14) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 14) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 14) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 15) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 15) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 15) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 16) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 16) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 16) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 17) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 17) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 17) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 18) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 18) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 18) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 19) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 19) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 19) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 20) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 20) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 20) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 21))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 21) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 21) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 21) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 22))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 22) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 22) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 22) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 23))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 23) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 23) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 23) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 24))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 24) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 24) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 24) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 25))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 25) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 25) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 25) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 26))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 26) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 26) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 26) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 27))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 27) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 27) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 27) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 28))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 28) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 28) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 28) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 29))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 29) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 29) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 29) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 30))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 30) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 30) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 30) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 31))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 31) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 31) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 31) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 32))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 32) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 32) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 32) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 32) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 33))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 33) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 33) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 33) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 33) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 34))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 34) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 34) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 34) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 34) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 35))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 35) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 35) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 35) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 35) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 36))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 36) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 36) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 36) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 36) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 37))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 37) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 37) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 37) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 37) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 38))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 38) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 38) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 38) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 38) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 39))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 39) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 39) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 39) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 39) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 40))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 40) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 40) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 40) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 40) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 41))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 41) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 41) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 41) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 41) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 42))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 42) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 42) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 42) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 42) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 43))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 43) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 43) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 43) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 43) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 44))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 44) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 44) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 44) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 44) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 45))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 45) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 45) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 45) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 45) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 46))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 46) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 46) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 46) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 46) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 47))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 47) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 47) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 47) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 47) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 48))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 48) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 48) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 48) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 48) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 49))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 49) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 49) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 49) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 49) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 50))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 50) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 50) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 50) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 50) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 51))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 51) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 51) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 51) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 51) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 52))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 52) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 52) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 52) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 52) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 53))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 53) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 53) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 53) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 53) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 54))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 54) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 54) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 54) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 54) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 55))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 55) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 55) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 55) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 55) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 56))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 56) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 56) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 56) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 56) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 57))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 57) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 57) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 57) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 57) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 58))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 58) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 58) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 58) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 58) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 59))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 59) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 59) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 59) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 59) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 60))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 60) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 60) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 60) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 60) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
    pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 61))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 61) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 61) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 61) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 61) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 62) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 62) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1986) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 62))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 62) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 62) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 62) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 62) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 63) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 63) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1985) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 63))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 63) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 63) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 63) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 63) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + ((((int)threadIdx.x) * 74) >> 6)) < 31) {
      if (((((int)threadIdx.z) * 64) + ((((int)threadIdx.x) * 74) >> 4)) < 124) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1984) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 64))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 74) & 63) >> 4)) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + (((((int)threadIdx.x) * 74) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + ((((((int)threadIdx.x) * 74) & 63) >> 4) * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 181))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 65) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 65) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1983) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 65))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 1) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 65) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 1) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 66) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 66) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1982) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 66))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 2) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 66) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 2) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 67) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 67) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1981) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 67))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 3) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 67) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 3) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 68) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 68) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1980) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 68))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 4) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 68) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 4) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 69) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 69) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1979) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 69))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 5) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 69) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 5) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 70) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 70) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1978) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 70))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 6) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 70) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 6) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 71) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 71) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1977) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 71))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 7) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 71) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 7) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 72) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 72) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1976) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 72))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 8) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 72) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 8) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 16) + (((((int)threadIdx.x) * 74) + 73) >> 6)) < 32) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 74) + 73) >> 4)) < 128) {
        if (((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) < 1975) {
          if (((int)threadIdx.x) < 13) {
            pad_temp_shared[((((((int)threadIdx.z) * 1024) + (((int)threadIdx.x) * 74)) + 73))] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 74) + 9) & 63) >> 4)) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[((((((((rc_outer * 6272) + (((int)threadIdx.z) * 3136)) + ((((((int)threadIdx.x) * 74) + 73) >> 6) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.x) * 74) + 9) & 63) >> 4) * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
          }
        }
      }
    }
    kernel_shared[(((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + (((((int)threadIdx.x) * 14) / 96) * 576)) + (rc_outer * 288)) + (((((int)threadIdx.x) * 14) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 1))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + (((((int)threadIdx.x) * 14) / 96) * 576)) + (rc_outer * 288)) + (((((int)threadIdx.x) * 14) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 2))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + (((((int)threadIdx.x) * 14) / 96) * 576)) + (rc_outer * 288)) + (((((int)threadIdx.x) * 14) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 3))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 1) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 1) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 4))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 1) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 1) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 5))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 1) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 1) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 6))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 2) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 2) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 7))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 2) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 2) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 8))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 2) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 2) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 9))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 3) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 3) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 10))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 3) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 3) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 11))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 3) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 3) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 12))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 4) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 4) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 13))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 4) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 4) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 14))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 4) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 4) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 15))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 5) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 5) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 16))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 5) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 5) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 17))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 5) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 5) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 18))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 6) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 6) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 19))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 6) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 6) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 20))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 6) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 6) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 21))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 7) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 7) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 22))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 7) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 7) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 23))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 7) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 7) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 24))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 8) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 8) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 25))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 8) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 8) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 26))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 8) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 8) % 96) * 3)) + 2))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 27))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 9) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 9) % 96) * 3)))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 28))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 9) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 9) % 96) * 3)) + 1))];
    kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 29))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 9) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 9) % 96) * 3)) + 2))];
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 10) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 10) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 374) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1122) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 30))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 10) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 10) % 96) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 10) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 10) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 374) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1121) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 31))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 10) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 10) % 96) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 10) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 10) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 374) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1120) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 32))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 10) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 10) % 96) * 3)) + 2))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 11) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 11) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 373) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1119) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 33))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 11) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 11) % 96) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 11) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 11) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 373) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1118) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 34))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 11) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 11) % 96) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 11) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 11) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 373) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1117) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 35))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 11) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 11) % 96) * 3)) + 2))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 12) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1116) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 36))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 12) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 12) % 96) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 12) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1115) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 37))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 12) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 12) % 96) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 12) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + ((((int)threadIdx.x) * 14) / 3)) < 124) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 372) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1114) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 38))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 12) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 12) % 96) * 3)) + 2))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 13) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1113) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 39))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 13) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 13) % 96) * 3)))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 13) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1112) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 40))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 13) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 13) % 96) * 3)) + 1))];
            }
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 14) + 13) / 96)) < 4) {
      if (((((int)threadIdx.z) * 64) + (((((int)threadIdx.x) * 14) + 13) / 3)) < 128) {
        if (((((int)threadIdx.z) * 192) + (((int)threadIdx.x) * 14)) < 371) {
          if (((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) < 1111) {
            if (((int)threadIdx.x) < 13) {
              kernel_shared[((((((int)threadIdx.z) * 576) + (((int)threadIdx.x) * 42)) + 41))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 14) + 13) / 96) * 576)) + (rc_outer * 288)) + ((((((int)threadIdx.x) * 14) + 13) % 96) * 3)) + 2))];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner_outer = 0; rc_inner_outer < 16; ++rc_inner_outer) {
      pad_temp_shared_local[(0)] = pad_temp_shared[(((rc_inner_outer * 128) + ((int)threadIdx.x)))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 16))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 32))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 48))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 64))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 80))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 96))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 112))];
      kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 576))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 3))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 579))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 6))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 582))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 9))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 585))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 12))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 588))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 15))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 591))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 1))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 17))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 33))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 49))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 65))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 81))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 97))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 113))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 1))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 577))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 4))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 580))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 7))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 583))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 10))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 586))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 13))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 589))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 16))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 592))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(11)]));
      pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 2))];
      pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 18))];
      pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 34))];
      pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 50))];
      pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 66))];
      pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 82))];
      pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 98))];
      pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 128) + ((int)threadIdx.x)) + 114))];
      kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 2))];
      kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 578))];
      kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 5))];
      kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 581))];
      kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 8))];
      kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 584))];
      kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 11))];
      kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 587))];
      kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 14))];
      kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 590))];
      kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 17))];
      kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 18)) + 593))];
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(7)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(8)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(8)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(9)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(9)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(10)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(10)]));
      compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(5)]));
      compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(11)]));
      compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
      compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(11)]));
    }
  }
  compute[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 392))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 406))] = compute_local[(3)];
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

    dim3 grid(1,7,8);
    dim3 block(14,1,2);

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


