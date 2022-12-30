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
  float compute_local[32];
  __shared__ float pad_temp_shared[280];
  __shared__ float kernel_shared[192];
  float pad_temp_shared_local[64];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 280) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)))] = ((((1 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14))) && (((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 14)))) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 14)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 279) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1))] = ((((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 14)))) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 14)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 278) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2))] = ((((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14)) < 57)) && (1 <= ((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 14)))) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 14)) - 57))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)))] = kernel[((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 32) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1))] = kernel[((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 576)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) * 2))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(48)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(49)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 14))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(50)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(51)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(52)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 112))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(53)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 113))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(54)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 126))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(55)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 127))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 140))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(56)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 141))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(57)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 154))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(58)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 155))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(59)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(60)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 252))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(61)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 253))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(62)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 266))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(63)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 267))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 6))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 6) + 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 6) + 1))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 6) + 97))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 6) + 2))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 6) + 98))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 6) + 3))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 6) + 99))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 6) + 4))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 6) + 100))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 6) + 5))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 6) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(6)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(0)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(6)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(0)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(6)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(0)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(0)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(6)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(0)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(6)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(0)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(6)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(0)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(6)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(0)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(1)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(7)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(1)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(1)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(1)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(7)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(1)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(7)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(1)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(7)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(1)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(7)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(1)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(7)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(1)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(7)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(1)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(2)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(8)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(2)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(8)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(2)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(2)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(8)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(2)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(8)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(2)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(8)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(2)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(2)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(8)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(2)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(8)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(2)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(2)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(8)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(2)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(9)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(3)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(9)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(3)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(9)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(3)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(3)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(9)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(3)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(9)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(3)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(3)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(9)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(3)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(3)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(3)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(9)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(3)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(3)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(4)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(10)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(4)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(10)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(4)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(4)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(10)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(4)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(10)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(4)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(4)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(4)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(4)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(10)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(4)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(4)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(4)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(10)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(4)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(4)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(11)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(5)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(11)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(5)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(5)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(11)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(5)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(11)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(5)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(11)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(5)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(5)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(11)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(5)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(5)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(5)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(5)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(11)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(5)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(5)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(11)]));
    __syncthreads();
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 280) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)))] = (((1 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14))) && (((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14)) < 57)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 14)) - 56))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 279) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14)) < 57)) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 14)) - 56))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 278) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14)) < 57)) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 14)) - 56))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)))] = kernel[(((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)) + 1))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 32) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1))] = kernel[(((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 576)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)) + 1))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) * 2))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(48)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(49)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 14))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(50)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(51)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(52)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 112))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(53)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 113))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(54)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 126))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(55)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 127))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 140))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(56)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 141))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(57)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 154))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(58)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 155))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(59)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(60)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 252))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(61)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 253))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(62)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 266))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(63)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 267))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 6))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 6) + 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 6) + 1))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 6) + 97))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 6) + 2))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 6) + 98))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 6) + 3))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 6) + 99))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 6) + 4))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 6) + 100))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 6) + 5))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 6) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(6)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(0)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(6)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(0)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(6)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(0)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(0)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(6)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(0)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(6)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(0)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(6)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(0)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(6)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(0)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(1)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(7)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(1)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(1)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(1)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(7)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(1)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(7)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(1)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(7)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(1)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(7)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(1)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(7)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(1)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(7)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(1)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(2)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(8)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(2)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(8)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(2)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(2)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(8)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(2)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(8)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(2)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(8)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(2)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(2)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(8)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(2)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(8)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(2)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(2)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(8)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(2)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(9)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(3)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(9)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(3)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(9)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(3)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(3)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(9)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(3)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(9)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(3)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(3)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(9)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(3)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(3)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(3)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(9)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(3)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(3)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(4)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(10)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(4)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(10)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(4)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(4)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(10)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(4)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(10)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(4)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(4)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(4)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(4)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(10)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(4)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(4)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(4)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(10)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(4)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(4)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(11)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(5)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(11)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(5)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(5)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(11)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(5)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(11)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(5)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(11)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(5)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(5)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(11)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(5)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(5)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(5)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(5)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(11)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(5)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(5)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(11)]));
    __syncthreads();
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 280) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)))] = ((((1 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14))) && (((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14)) < 57)) && (((((int)blockIdx.x) * 14) + (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 14)) < 55)) ? data[((((((((rc_outer * 6272) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) % 14)) - 55))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 279) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1))] = ((((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14)) < 57)) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 14)) < 55)) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 1) % 14)) - 55))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) < 278) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2))] = ((((1 <= ((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14))) && (((((int)blockIdx.y) * 8) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14)) < 57)) && (((((int)blockIdx.x) * 14) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 14)) < 55)) ? data[((((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) / 140) * 3136)) + (((int)blockIdx.y) * 448)) + ((((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 140) / 14) * 56)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 18) + (((int)threadIdx.x) * 3)) + 2) % 14)) - 55))] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 2) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)))] = kernel[(((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 3) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 3) * 6)) + 2))];
          }
        }
      }
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.x) * 2) + 1) / 6)) < 32) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
          if (((int)threadIdx.x) < 6) {
            kernel_shared[((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1))] = kernel[(((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((((int)threadIdx.x) * 2) + 1) / 6) * 576)) + (rc_outer * 18)) + ((((((int)threadIdx.x) * 2) + 1) % 6) * 3)) + 2))];
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[((((int)threadIdx.x) * 2))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(48)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(49)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 14))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(50)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(51)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 28))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 56))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 84))];
    pad_temp_shared_local[(52)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 112))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 29))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 57))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 85))];
    pad_temp_shared_local[(53)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 113))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 42))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 70))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 98))];
    pad_temp_shared_local[(54)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 126))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 43))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 71))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 99))];
    pad_temp_shared_local[(55)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 127))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 140))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(56)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 141))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(57)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 154))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(58)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 155))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(59)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 168))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 196))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 224))];
    pad_temp_shared_local[(60)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 252))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 197))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 225))];
    pad_temp_shared_local[(61)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 253))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 182))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 210))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 238))];
    pad_temp_shared_local[(62)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 266))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 183))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 211))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 239))];
    pad_temp_shared_local[(63)] = pad_temp_shared[(((((int)threadIdx.x) * 2) + 267))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 6))];
    kernel_shared_local[(6)] = kernel_shared[(((((int)threadIdx.z) * 6) + 96))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 6) + 1))];
    kernel_shared_local[(7)] = kernel_shared[(((((int)threadIdx.z) * 6) + 97))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 6) + 2))];
    kernel_shared_local[(8)] = kernel_shared[(((((int)threadIdx.z) * 6) + 98))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 6) + 3))];
    kernel_shared_local[(9)] = kernel_shared[(((((int)threadIdx.z) * 6) + 99))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 6) + 4))];
    kernel_shared_local[(10)] = kernel_shared[(((((int)threadIdx.z) * 6) + 100))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 6) + 5))];
    kernel_shared_local[(11)] = kernel_shared[(((((int)threadIdx.z) * 6) + 101))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(6)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(6)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(0)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(6)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(0)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(6)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(6)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(0)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(0)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(6)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(6)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(0)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(6)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(0)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(6)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(6)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(0)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(6)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(0)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(1)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(7)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(1)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(7)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(1)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(1)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(1)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(7)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(1)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(7)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(1)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(7)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(1)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(7)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(7)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(1)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(7)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(1)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(7)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(7)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(1)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(7)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(1)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(2)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(8)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(2)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(8)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(2)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(8)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(2)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(8)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(2)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(8)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(2)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(8)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(2)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(2)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(8)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(2)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(8)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(2)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(8)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(2)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(2)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(8)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(2)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(8)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(2)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(8)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(2)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(3)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(9)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(3)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(9)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(3)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(9)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(3)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(3)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(3)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(9)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(3)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(9)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(3)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(3)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(9)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(3)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(9)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(3)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(3)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(3)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(9)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(3)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(9)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(3)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(9)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(3)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(4)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(4)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(10)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(4)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(10)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(4)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(4)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(10)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(4)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(10)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(4)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(4)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(4)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(10)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(4)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(10)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(4)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(10)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(4)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(10)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(4)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(10)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(4)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(10)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(4)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(5)]));
    compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(11)]));
    compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(5)]));
    compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(11)]));
    compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(5)]));
    compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(5)]));
    compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(5)]));
    compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(11)]));
    compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(5)]));
    compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(11)]));
    compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(5)]));
    compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(11)]));
    compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(5)]));
    compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(11)]));
    compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(5)]));
    compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(11)]));
    compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(5)]));
    compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(11)]));
    compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(5)]));
    compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(5)]));
    compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(11)]));
    compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(5)]));
    compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(11)]));
    compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(5)]));
    compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(5)]));
    compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(11)]));
  }
  compute[((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)))] = compute_local[(0)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50176))] = compute_local[(16)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 112))] = compute_local[(4)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50288))] = compute_local[(20)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 224))] = compute_local[(8)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50400))] = compute_local[(24)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 336))] = compute_local[(12)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50512))] = compute_local[(28)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 1))] = compute_local[(1)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50177))] = compute_local[(17)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 113))] = compute_local[(5)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50289))] = compute_local[(21)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 225))] = compute_local[(9)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50401))] = compute_local[(25)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 337))] = compute_local[(13)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50513))] = compute_local[(29)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 56))] = compute_local[(2)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50232))] = compute_local[(18)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 168))] = compute_local[(6)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50344))] = compute_local[(22)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 280))] = compute_local[(10)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50456))] = compute_local[(26)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 392))] = compute_local[(14)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50568))] = compute_local[(30)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 57))] = compute_local[(3)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50233))] = compute_local[(19)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 169))] = compute_local[(7)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50345))] = compute_local[(23)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 281))] = compute_local[(11)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50457))] = compute_local[(27)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 393))] = compute_local[(15)];
  compute[(((((((((int)blockIdx.z) * 100352) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 14)) + (((int)threadIdx.x) * 2)) + 50569))] = compute_local[(31)];
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


