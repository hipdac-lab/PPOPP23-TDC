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
  float compute_local[28];
  __shared__ float pad_temp_shared[1856];
  __shared__ float kernel_shared[1536];
  float pad_temp_shared_local[14];
  float kernel_shared_local[8];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
      __syncthreads();
      pad_temp_shared[((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)))] = (((((2 <= (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) % 116)) && ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) / 116) * 3136)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 1))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 2))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 2) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 2) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 2) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 2) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 3))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 3) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 3) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 3) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 3) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 4))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 4) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 4) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 4) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 4) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 5))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 5) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 5) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 5) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 5) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 6))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 6) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 6) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 6) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 6) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 7))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 7) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 7) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 7) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 7) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 8))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 8) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 8) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 8) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 8) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 9))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 9) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 9) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 9) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 9) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 10))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 10) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 10) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 10) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 10) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 11))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 11) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 11) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 11) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 11) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 12))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 12) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 12) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 12) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 12) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 13))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 13) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 13) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 13) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 13) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 1) & 1)) - 57))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 2) + ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 14) / 116)) < 16) {
        if (((((int)threadIdx.z) * 116) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) >> 1)) < 921) {
          if ((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) < 1842) {
            if (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) < 218) {
              if (((int)threadIdx.x) < 1) {
                pad_temp_shared[(((((((int)threadIdx.z) * 232) + (((int)threadIdx.y) * 29)) + (((int)threadIdx.x) * 15)) + 14))] = (((((2 <= ((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 14) % 116)) && (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 14) % 116) < 114)) && (1 <= (((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)))) && ((((((int)blockIdx.x) * 2) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) < 57)) ? data[(((((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 14) / 116) * 3136)) + ((((((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) + 14) % 116) >> 1) * 56)) + (((int)blockIdx.x) * 2)) + rx_outer) + (((((int)threadIdx.y) * 29) + (((int)threadIdx.x) * 15)) & 1)) - 57))] : 0.000000e+00f);
              }
            }
          }
        }
      }
      kernel_shared[((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)))] = kernel[((((((((int)threadIdx.z) * 2304) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 576)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15) * 9)) + rx_outer))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 1))] = kernel[(((((((((int)threadIdx.z) * 2304) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 576)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15) * 9)) + rx_outer) + 3))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 2))] = kernel[(((((((((int)threadIdx.z) * 2304) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) >> 4) * 576)) + (rc_outer * 144)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) & 15) * 9)) + rx_outer) + 6))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 3))] = kernel[((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15) * 9)) + rx_outer))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 4))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15) * 9)) + rx_outer) + 3))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 5))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 1) & 15) * 9)) + rx_outer) + 6))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 6))] = kernel[((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15) * 9)) + rx_outer))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 7))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15) * 9)) + rx_outer) + 3))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 8))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 2) & 15) * 9)) + rx_outer) + 6))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 9))] = kernel[((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15) * 9)) + rx_outer))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 10))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15) * 9)) + rx_outer) + 3))];
      kernel_shared[(((((((int)threadIdx.z) * 192) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 12)) + 11))] = kernel[(((((((((int)threadIdx.z) * 2304) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) >> 4) * 576)) + (rc_outer * 144)) + (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 4)) + 3) & 15) * 9)) + rx_outer) + 6))];
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
        pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 116))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 118))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 120))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 122))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 124))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 126))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 128))];
        kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 3))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 48))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 51))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 96))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 99))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 144))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 147))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(0)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(4)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(4)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(6)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(6)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(1)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(1)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(1)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(5)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(7)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(7)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(7)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 14))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 118))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 120))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 122))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 124))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 126))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 128))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 130))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 1))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 4))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 49))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 52))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 97))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 100))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 145))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 148))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(0)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(4)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(4)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(6)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(6)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(1)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(1)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(1)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(5)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(7)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(7)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(7)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 4))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 6))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 8))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 10))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 12))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 14))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 16))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 120))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 122))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 124))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 126))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 128))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 130))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 232) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 132))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 2))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 5))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 50))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 53))];
        kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 98))];
        kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 101))];
        kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 146))];
        kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 192) + (rc_inner_outer * 6)) + 149))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(0)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(0)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(4)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(4)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(4)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(4)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(4)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(4)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(6)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(6)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(1)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(1)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(1)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(5)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(5)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(5)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(7)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(7)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(7)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(7)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
      }
    }
  }
  compute[(((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 56))] = compute_local[(1)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 112))] = compute_local[(2)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 168))] = compute_local[(3)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 224))] = compute_local[(4)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 280))] = compute_local[(5)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 336))] = compute_local[(6)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3136))] = compute_local[(7)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3192))] = compute_local[(8)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3248))] = compute_local[(9)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3304))] = compute_local[(10)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3360))] = compute_local[(11)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3416))] = compute_local[(12)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 3472))] = compute_local[(13)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6272))] = compute_local[(14)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6328))] = compute_local[(15)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6384))] = compute_local[(16)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6440))] = compute_local[(17)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6496))] = compute_local[(18)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6552))] = compute_local[(19)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 6608))] = compute_local[(20)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9408))] = compute_local[(21)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9464))] = compute_local[(22)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9520))] = compute_local[(23)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9576))] = compute_local[(24)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9632))] = compute_local[(25)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9688))] = compute_local[(26)];
  compute[((((((((int)threadIdx.z) * 12544) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 9744))] = compute_local[(27)];
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

    dim3 grid(28,1,1);
    dim3 block(2,8,8);

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


