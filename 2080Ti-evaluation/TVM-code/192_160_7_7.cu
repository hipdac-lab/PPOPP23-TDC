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

#define C 192
#define N 160
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
  __shared__ float pad_temp_shared[196];
  __shared__ float kernel_shared[20];
  float pad_temp_shared_local[1];
  float kernel_shared_local[1];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 48; ++rc_outer) {
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = (((7 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49)) && (1 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7))) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) - 8))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = ((7 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49)) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) - 7))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = (((7 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49)) && (((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7) < 6)) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) - 6))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = ((1 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7)) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) - 1))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 3))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = data[(((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))];
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 4))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = ((((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7) < 6) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) + 1))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 5))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = (((((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49) < 42) && (1 <= ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7))) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) + 6))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 6))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = ((((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49) < 42) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) + 7))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 7))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    __syncthreads();
    if ((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) < 196) {
      if (((((int)threadIdx.y) * 6) + ((int)threadIdx.x)) < 40) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)))] = (((((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 49) < 42) && (((((((int)threadIdx.z) * 40) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) % 7) < 6)) ? data[((((((rc_outer * 196) + (((int)threadIdx.z) * 40)) + (((int)threadIdx.y) * 6)) + ((int)threadIdx.x)) + 8))] : 0.000000e+00f);
        }
      }
    }
    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) >> 2) + ((int)threadIdx.z)) < 5) {
      if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)) < 20) {
        if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 4) {
          if (((int)threadIdx.x) < 1) {
            kernel_shared[((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) + ((int)threadIdx.y)))] = kernel[(((((((((int)blockIdx.z) * 8640) + (((int)threadIdx.z) * 1728)) + (rc_outer * 36)) + (((int)threadIdx.x) * 9)) + (((int)threadIdx.y) * 9)) + 8))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 4))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 1))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 2))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 4) + 3))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
  }
  compute[(((((((int)blockIdx.z) * 245) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
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

    dim3 grid(1,1,32);
    dim3 block(7,7,5);

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

