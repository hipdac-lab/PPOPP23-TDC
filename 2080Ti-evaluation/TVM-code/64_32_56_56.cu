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
  float compute_local[7];
  __shared__ float pad_temp_shared[928];
  __shared__ float kernel_shared[288];
  float pad_temp_shared_local[54];
  float kernel_shared_local[18];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + ((((int)threadIdx.y) * 15) & 3)))) && (((((int)blockIdx.x) * 2) + ((((int)threadIdx.y) * 15) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + ((((int)threadIdx.y) * 15) & 3)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 1))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 1) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 1) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) & 3)) - 57))] : 0.000000e+00f);
    pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 2))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 2) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 2) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) & 3)) - 57))] : 0.000000e+00f);
    if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) >> 2)) < 232) {
      if ((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) < 925) {
        if (((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 8)) < 113) {
          pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 3))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 3) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 3) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) & 3)) - 57))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) < 231) {
      if ((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) < 924) {
        if (((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 8)) < 112) {
          pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 4))] = (((((1 <= (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) + 1) % 58)) && ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) + 1) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + ((((int)threadIdx.y) * 15) & 3)))) && (((((int)blockIdx.x) * 2) + ((((int)threadIdx.y) * 15) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) + 1) / 58) * 3136)) + ((((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ((((int)threadIdx.y) * 15) >> 2)) + 1) % 58) * 56)) + (((int)blockIdx.x) * 2)) + ((((int)threadIdx.y) * 15) & 3)) - 57))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 5) >> 2)) < 232) {
      if ((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) < 923) {
        if (((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 8)) < 111) {
          pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 5))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 5) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 5) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 1) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 1) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 5) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 5) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 1) & 3)) - 57))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 6) >> 2)) < 232) {
      if ((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) < 922) {
        if (((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 8)) < 110) {
          pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 6))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 6) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 6) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 2) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 2) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 6) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 6) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 2) & 3)) - 57))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 7) >> 2)) < 232) {
      if ((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) < 921) {
        if (((((int)threadIdx.y) * 15) + (((int)threadIdx.x) * 8)) < 109) {
          if (((int)threadIdx.x) < 1) {
            pad_temp_shared[(((((((int)threadIdx.z) * 116) + (((int)threadIdx.y) * 15)) + (((int)threadIdx.x) * 8)) + 7))] = (((((1 <= ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 7) >> 2)) % 58)) && (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 7) >> 2)) % 58) < 57)) && (1 <= ((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 3) & 3)))) && (((((int)blockIdx.x) * 2) + (((((int)threadIdx.y) * 15) + 3) & 3)) < 57)) ? data[(((((((rc_outer * 12544) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 7) >> 2)) / 58) * 3136)) + (((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 7) >> 2)) % 58) * 56)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.y) * 15) + 3) & 3)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
    if ((((((((int)threadIdx.y) * 5) / 3) + ((int)threadIdx.x)) / 12) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 5) / 3) + ((int)threadIdx.x)) / 3)) < 32) {
        if ((((((int)threadIdx.z) * 12) + ((((int)threadIdx.y) * 5) / 3)) + ((int)threadIdx.x)) < 96) {
          if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) < 288) {
            if (((((int)threadIdx.y) * 5) + (((int)threadIdx.x) * 3)) < 36) {
              if ((((((int)blockIdx.z) * 8) + ((((((int)threadIdx.y) * 5) / 3) + ((int)threadIdx.x)) / 12)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)))] = kernel[(((((((((int)blockIdx.z) * 4608) + (((((((int)threadIdx.y) * 5) / 3) + ((int)threadIdx.x)) / 12) * 576)) + (((int)threadIdx.z) * 576)) + (rc_outer * 36)) + (((((((int)threadIdx.y) * 5) / 3) + ((int)threadIdx.x)) % 12) * 3)) + ((((int)threadIdx.y) * 5) % 3)))];
              }
            }
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 5) + 1) / 3) + ((int)threadIdx.x)) / 12) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 4) + (((((((int)threadIdx.y) * 5) + 1) / 3) + ((int)threadIdx.x)) / 3)) < 32) {
        if ((((((int)threadIdx.z) * 12) + (((((int)threadIdx.y) * 5) + 1) / 3)) + ((int)threadIdx.x)) < 96) {
          if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) < 287) {
            if (((((int)threadIdx.y) * 5) + (((int)threadIdx.x) * 3)) < 35) {
              if ((((((int)blockIdx.z) * 8) + (((((((int)threadIdx.y) * 5) + 1) / 3) + ((int)threadIdx.x)) / 12)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + 1))] = kernel[(((((((((int)blockIdx.z) * 4608) + ((((((((int)threadIdx.y) * 5) + 1) / 3) + ((int)threadIdx.x)) / 12) * 576)) + (((int)threadIdx.z) * 576)) + (rc_outer * 36)) + ((((((((int)threadIdx.y) * 5) + 1) / 3) + ((int)threadIdx.x)) % 12) * 3)) + (((((int)threadIdx.y) * 5) + 1) % 3)))];
              }
            }
          }
        }
      }
    }
    if (((((((((int)threadIdx.y) * 5) + 2) / 3) + ((int)threadIdx.x)) / 12) + ((int)threadIdx.z)) < 8) {
      if (((((int)threadIdx.z) * 4) + (((((((int)threadIdx.y) * 5) + 2) / 3) + ((int)threadIdx.x)) / 3)) < 32) {
        if ((((((int)threadIdx.z) * 12) + (((((int)threadIdx.y) * 5) + 2) / 3)) + ((int)threadIdx.x)) < 96) {
          if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) < 286) {
            if (((((int)threadIdx.y) * 5) + (((int)threadIdx.x) * 3)) < 34) {
              if (((int)threadIdx.x) < 1) {
                if ((((((int)blockIdx.z) * 8) + (((((((int)threadIdx.y) * 5) + 2) / 3) + ((int)threadIdx.x)) / 12)) + ((int)threadIdx.z)) < 32) {
                  kernel_shared[(((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 5)) + (((int)threadIdx.x) * 3)) + 2))] = kernel[(((((((((int)blockIdx.z) * 4608) + ((((((((int)threadIdx.y) * 5) + 2) / 3) + ((int)threadIdx.x)) / 12) * 576)) + (((int)threadIdx.z) * 576)) + (rc_outer * 36)) + ((((((((int)threadIdx.y) * 5) + 2) / 3) + ((int)threadIdx.x)) % 12) * 3)) + (((((int)threadIdx.y) * 5) + 2) % 3)))];
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 5))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 6))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 8))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 12))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 13))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 14))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 16))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 17))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 24))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 25))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 26))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 28))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 29))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 30))];
    pad_temp_shared_local[(24)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 32))];
    pad_temp_shared_local[(25)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 33))];
    pad_temp_shared_local[(26)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 34))];
    pad_temp_shared_local[(27)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 232))];
    pad_temp_shared_local[(28)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 233))];
    pad_temp_shared_local[(29)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 234))];
    pad_temp_shared_local[(30)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 236))];
    pad_temp_shared_local[(31)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 237))];
    pad_temp_shared_local[(32)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 238))];
    pad_temp_shared_local[(33)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 240))];
    pad_temp_shared_local[(34)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 241))];
    pad_temp_shared_local[(35)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 242))];
    pad_temp_shared_local[(36)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 244))];
    pad_temp_shared_local[(37)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 245))];
    pad_temp_shared_local[(38)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 246))];
    pad_temp_shared_local[(39)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 248))];
    pad_temp_shared_local[(40)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 249))];
    pad_temp_shared_local[(41)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 250))];
    pad_temp_shared_local[(42)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 252))];
    pad_temp_shared_local[(43)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 253))];
    pad_temp_shared_local[(44)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 254))];
    pad_temp_shared_local[(45)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 256))];
    pad_temp_shared_local[(46)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 257))];
    pad_temp_shared_local[(47)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 258))];
    pad_temp_shared_local[(48)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 260))];
    pad_temp_shared_local[(49)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 261))];
    pad_temp_shared_local[(50)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 262))];
    pad_temp_shared_local[(51)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 264))];
    pad_temp_shared_local[(52)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 265))];
    pad_temp_shared_local[(53)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 266))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 1))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 2))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 3))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 4))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 5))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 6))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 7))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 8))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 9))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 10))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 11))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 12))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 13))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 14))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 15))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 16))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 17))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(0)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(1)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(2)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(3)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(4)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(5)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(9)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(9)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(10)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(10)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(11)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(11)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(12)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(12)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(12)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(13)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(13)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(13)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(14)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(14)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(14)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(15)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(15)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(15)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(16)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(16)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(16)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(17)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(17)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(17)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(17)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 464))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 465))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 466))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 468))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 469))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 470))];
    pad_temp_shared_local[(6)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 472))];
    pad_temp_shared_local[(7)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 473))];
    pad_temp_shared_local[(8)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 474))];
    pad_temp_shared_local[(9)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 476))];
    pad_temp_shared_local[(10)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 477))];
    pad_temp_shared_local[(11)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 478))];
    pad_temp_shared_local[(12)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 480))];
    pad_temp_shared_local[(13)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 481))];
    pad_temp_shared_local[(14)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 482))];
    pad_temp_shared_local[(15)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 484))];
    pad_temp_shared_local[(16)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 485))];
    pad_temp_shared_local[(17)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 486))];
    pad_temp_shared_local[(18)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 488))];
    pad_temp_shared_local[(19)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 489))];
    pad_temp_shared_local[(20)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 490))];
    pad_temp_shared_local[(21)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 492))];
    pad_temp_shared_local[(22)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 493))];
    pad_temp_shared_local[(23)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 494))];
    pad_temp_shared_local[(24)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 496))];
    pad_temp_shared_local[(25)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 497))];
    pad_temp_shared_local[(26)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 498))];
    pad_temp_shared_local[(27)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 696))];
    pad_temp_shared_local[(28)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 697))];
    pad_temp_shared_local[(29)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 698))];
    pad_temp_shared_local[(30)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 700))];
    pad_temp_shared_local[(31)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 701))];
    pad_temp_shared_local[(32)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 702))];
    pad_temp_shared_local[(33)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 704))];
    pad_temp_shared_local[(34)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 705))];
    pad_temp_shared_local[(35)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 706))];
    pad_temp_shared_local[(36)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 708))];
    pad_temp_shared_local[(37)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 709))];
    pad_temp_shared_local[(38)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 710))];
    pad_temp_shared_local[(39)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 712))];
    pad_temp_shared_local[(40)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 713))];
    pad_temp_shared_local[(41)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 714))];
    pad_temp_shared_local[(42)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 716))];
    pad_temp_shared_local[(43)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 717))];
    pad_temp_shared_local[(44)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 718))];
    pad_temp_shared_local[(45)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 720))];
    pad_temp_shared_local[(46)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 721))];
    pad_temp_shared_local[(47)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 722))];
    pad_temp_shared_local[(48)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 724))];
    pad_temp_shared_local[(49)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 725))];
    pad_temp_shared_local[(50)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 726))];
    pad_temp_shared_local[(51)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 728))];
    pad_temp_shared_local[(52)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 729))];
    pad_temp_shared_local[(53)] = pad_temp_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 730))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 36) + 18))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 36) + 19))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 36) + 20))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 36) + 21))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 36) + 22))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 36) + 23))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 36) + 24))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 36) + 25))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 36) + 26))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 36) + 27))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 36) + 28))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 36) + 29))];
    kernel_shared_local[(12)] = kernel_shared[(((((int)threadIdx.z) * 36) + 30))];
    kernel_shared_local[(13)] = kernel_shared[(((((int)threadIdx.z) * 36) + 31))];
    kernel_shared_local[(14)] = kernel_shared[(((((int)threadIdx.z) * 36) + 32))];
    kernel_shared_local[(15)] = kernel_shared[(((((int)threadIdx.z) * 36) + 33))];
    kernel_shared_local[(16)] = kernel_shared[(((((int)threadIdx.z) * 36) + 34))];
    kernel_shared_local[(17)] = kernel_shared[(((((int)threadIdx.z) * 36) + 35))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(0)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(0)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(1)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(2)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(2)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(3)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(4)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(4)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(5)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(5)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(5)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(6)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(7)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(9)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(9)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(10)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(10)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(11)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(11)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(12)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(12)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(12)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(12)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(12)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(12)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(12)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(13)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(13)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(13)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(13)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(13)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(13)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(13)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(14)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(14)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(14)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(14)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(14)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(14)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(14)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(15)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(15)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(15)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(15)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(15)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(15)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(15)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(16)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(16)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(16)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(16)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(16)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(16)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(16)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(17)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(17)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(17)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(17)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(17)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(17)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(17)]));
  }
  compute[((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 56))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 112))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 168))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 224))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 280))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 336))] = compute_local[(6)];
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

    dim3 grid(28,1,4);
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


