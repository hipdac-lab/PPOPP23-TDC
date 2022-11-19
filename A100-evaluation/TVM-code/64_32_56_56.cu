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
  float compute_local[32];
  __shared__ float pad_temp_shared[384];
  __shared__ float kernel_shared[768];
  float pad_temp_shared_local[32];
  float kernel_shared_local[4];
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
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.z) * 48))] = (((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) && (1 <= ((int)blockIdx.x))) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 57))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 1))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 56))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 2))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 55))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 3))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 54))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 4))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 53))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 5))] = (((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) && (((int)blockIdx.x) < 13)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 52))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 6))] = ((1 <= ((int)blockIdx.x)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) - 1))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 7))] = data[((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 8))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 1))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 9))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 2))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 10))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 11))] = ((((int)blockIdx.x) < 13) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 4))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 12))] = ((1 <= ((int)blockIdx.x)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 55))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 13))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 56))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 14))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 57))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 15))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 58))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 16))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 59))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 17))] = ((((int)blockIdx.x) < 13) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 60))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 18))] = (((((((int)blockIdx.y) * 4) + ry_outer) < 54) && (1 <= ((int)blockIdx.x))) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 111))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 19))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 112))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 20))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 113))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 21))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 114))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 22))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 115))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 23))] = (((((((int)blockIdx.y) * 4) + ry_outer) < 54) && (((int)blockIdx.x) < 13)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 116))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 24))] = (((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) && (1 <= ((int)blockIdx.x))) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3079))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 25))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3080))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 26))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3081))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 27))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3082))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 28))] = ((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3083))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 29))] = (((1 <= ((((int)blockIdx.y) * 4) + ry_outer)) && (((int)blockIdx.x) < 13)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3084))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 30))] = ((1 <= ((int)blockIdx.x)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3135))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 31))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3136))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 32))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3137))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 33))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3138))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 34))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3139))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 35))] = ((((int)blockIdx.x) < 13) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3140))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 36))] = ((1 <= ((int)blockIdx.x)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3191))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 37))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3192))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 38))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3193))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 39))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3194))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 40))] = data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3195))];
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 41))] = ((((int)blockIdx.x) < 13) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3196))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 42))] = (((((((int)blockIdx.y) * 4) + ry_outer) < 54) && (1 <= ((int)blockIdx.x))) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3247))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 43))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3248))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 44))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3249))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 45))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3250))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 46))] = ((((((int)blockIdx.y) * 4) + ry_outer) < 54) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3251))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.z) * 48) + 47))] = (((((((int)blockIdx.y) * 4) + ry_outer) < 54) && (((int)blockIdx.x) < 13)) ? data[(((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + (((int)blockIdx.y) * 224)) + (ry_outer * 56)) + (((int)blockIdx.x) * 4)) + 3252))] : 0.000000e+00f);
      kernel_shared[((((int)threadIdx.z) * 96))] = kernel[(((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 1))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 1))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 2))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 2))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 3))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 9))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 4))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 10))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 5))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 11))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 6))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 18))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 7))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 19))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 8))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 20))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 9))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 27))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 10))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 28))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 11))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 29))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 12))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 36))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 13))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 37))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 14))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 38))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 15))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 45))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 16))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 46))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 17))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 47))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 18))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 54))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 19))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 55))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 20))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 56))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 21))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 63))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 22))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 64))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 23))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 65))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 24))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 72))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 25))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 73))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 26))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 74))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 27))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 81))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 28))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 82))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 29))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 83))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 30))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 90))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 31))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 91))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 32))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 92))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 33))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 99))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 34))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 100))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 35))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 101))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 36))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 108))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 37))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 109))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 38))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 110))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 39))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 117))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 40))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 118))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 41))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 119))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 42))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 126))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 43))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 127))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 44))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 128))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 45))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 135))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 46))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 136))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 47))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 137))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 48))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 576))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 49))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 577))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 50))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 578))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 51))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 585))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 52))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 586))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 53))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 587))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 54))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 594))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 55))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 595))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 56))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 596))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 57))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 603))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 58))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 604))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 59))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 605))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 60))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 612))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 61))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 613))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 62))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 614))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 63))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 621))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 64))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 622))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 65))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 623))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 66))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 630))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 67))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 631))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 68))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 632))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 69))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 639))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 70))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 640))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 71))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 641))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 72))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 648))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 73))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 649))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 74))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 650))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 75))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 657))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 76))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 658))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 77))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 659))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 78))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 666))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 79))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 667))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 80))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 668))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 81))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 675))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 82))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 676))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 83))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 677))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 84))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 684))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 85))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 685))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 86))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 686))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 87))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 693))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 88))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 694))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 89))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 695))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 90))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 702))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 91))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 703))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 92))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 704))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 93))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 711))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 94))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 712))];
      kernel_shared[(((((int)threadIdx.z) * 96) + 95))] = kernel[((((((((int)blockIdx.z) * 9216) + (((int)threadIdx.z) * 1152)) + (rc_outer * 144)) + (ry_outer * 3)) + 713))];
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
        pad_temp_shared_local[(0)] = pad_temp_shared[((rc_inner_outer * 48))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((rc_inner_outer * 48) + 1))];
        pad_temp_shared_local[(16)] = pad_temp_shared[(((rc_inner_outer * 48) + 2))];
        pad_temp_shared_local[(24)] = pad_temp_shared[(((rc_inner_outer * 48) + 3))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((rc_inner_outer * 48) + 6))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((rc_inner_outer * 48) + 7))];
        pad_temp_shared_local[(17)] = pad_temp_shared[(((rc_inner_outer * 48) + 8))];
        pad_temp_shared_local[(25)] = pad_temp_shared[(((rc_inner_outer * 48) + 9))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((rc_inner_outer * 48) + 12))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((rc_inner_outer * 48) + 13))];
        pad_temp_shared_local[(18)] = pad_temp_shared[(((rc_inner_outer * 48) + 14))];
        pad_temp_shared_local[(26)] = pad_temp_shared[(((rc_inner_outer * 48) + 15))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((rc_inner_outer * 48) + 18))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((rc_inner_outer * 48) + 19))];
        pad_temp_shared_local[(19)] = pad_temp_shared[(((rc_inner_outer * 48) + 20))];
        pad_temp_shared_local[(27)] = pad_temp_shared[(((rc_inner_outer * 48) + 21))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((rc_inner_outer * 48) + 24))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((rc_inner_outer * 48) + 25))];
        pad_temp_shared_local[(20)] = pad_temp_shared[(((rc_inner_outer * 48) + 26))];
        pad_temp_shared_local[(28)] = pad_temp_shared[(((rc_inner_outer * 48) + 27))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((rc_inner_outer * 48) + 30))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((rc_inner_outer * 48) + 31))];
        pad_temp_shared_local[(21)] = pad_temp_shared[(((rc_inner_outer * 48) + 32))];
        pad_temp_shared_local[(29)] = pad_temp_shared[(((rc_inner_outer * 48) + 33))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((rc_inner_outer * 48) + 36))];
        pad_temp_shared_local[(14)] = pad_temp_shared[(((rc_inner_outer * 48) + 37))];
        pad_temp_shared_local[(22)] = pad_temp_shared[(((rc_inner_outer * 48) + 38))];
        pad_temp_shared_local[(30)] = pad_temp_shared[(((rc_inner_outer * 48) + 39))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((rc_inner_outer * 48) + 42))];
        pad_temp_shared_local[(15)] = pad_temp_shared[(((rc_inner_outer * 48) + 43))];
        pad_temp_shared_local[(23)] = pad_temp_shared[(((rc_inner_outer * 48) + 44))];
        pad_temp_shared_local[(31)] = pad_temp_shared[(((rc_inner_outer * 48) + 45))];
        kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 384))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 3))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 387))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(0)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(2)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(2)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(0)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(0)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(2)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(0)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(2)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(0)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(2)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(1)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(3)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(1)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(3)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(1)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(1)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(1)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(1)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(1)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((rc_inner_outer * 48) + 1))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((rc_inner_outer * 48) + 2))];
        pad_temp_shared_local[(16)] = pad_temp_shared[(((rc_inner_outer * 48) + 3))];
        pad_temp_shared_local[(24)] = pad_temp_shared[(((rc_inner_outer * 48) + 4))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((rc_inner_outer * 48) + 7))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((rc_inner_outer * 48) + 8))];
        pad_temp_shared_local[(17)] = pad_temp_shared[(((rc_inner_outer * 48) + 9))];
        pad_temp_shared_local[(25)] = pad_temp_shared[(((rc_inner_outer * 48) + 10))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((rc_inner_outer * 48) + 13))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((rc_inner_outer * 48) + 14))];
        pad_temp_shared_local[(18)] = pad_temp_shared[(((rc_inner_outer * 48) + 15))];
        pad_temp_shared_local[(26)] = pad_temp_shared[(((rc_inner_outer * 48) + 16))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((rc_inner_outer * 48) + 19))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((rc_inner_outer * 48) + 20))];
        pad_temp_shared_local[(19)] = pad_temp_shared[(((rc_inner_outer * 48) + 21))];
        pad_temp_shared_local[(27)] = pad_temp_shared[(((rc_inner_outer * 48) + 22))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((rc_inner_outer * 48) + 25))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((rc_inner_outer * 48) + 26))];
        pad_temp_shared_local[(20)] = pad_temp_shared[(((rc_inner_outer * 48) + 27))];
        pad_temp_shared_local[(28)] = pad_temp_shared[(((rc_inner_outer * 48) + 28))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((rc_inner_outer * 48) + 31))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((rc_inner_outer * 48) + 32))];
        pad_temp_shared_local[(21)] = pad_temp_shared[(((rc_inner_outer * 48) + 33))];
        pad_temp_shared_local[(29)] = pad_temp_shared[(((rc_inner_outer * 48) + 34))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((rc_inner_outer * 48) + 37))];
        pad_temp_shared_local[(14)] = pad_temp_shared[(((rc_inner_outer * 48) + 38))];
        pad_temp_shared_local[(22)] = pad_temp_shared[(((rc_inner_outer * 48) + 39))];
        pad_temp_shared_local[(30)] = pad_temp_shared[(((rc_inner_outer * 48) + 40))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((rc_inner_outer * 48) + 43))];
        pad_temp_shared_local[(15)] = pad_temp_shared[(((rc_inner_outer * 48) + 44))];
        pad_temp_shared_local[(23)] = pad_temp_shared[(((rc_inner_outer * 48) + 45))];
        pad_temp_shared_local[(31)] = pad_temp_shared[(((rc_inner_outer * 48) + 46))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 1))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 385))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 4))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 388))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(0)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(2)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(2)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(0)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(0)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(2)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(0)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(2)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(0)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(2)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(1)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(3)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(1)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(3)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(1)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(1)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(1)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(1)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(1)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
        pad_temp_shared_local[(0)] = pad_temp_shared[(((rc_inner_outer * 48) + 2))];
        pad_temp_shared_local[(8)] = pad_temp_shared[(((rc_inner_outer * 48) + 3))];
        pad_temp_shared_local[(16)] = pad_temp_shared[(((rc_inner_outer * 48) + 4))];
        pad_temp_shared_local[(24)] = pad_temp_shared[(((rc_inner_outer * 48) + 5))];
        pad_temp_shared_local[(1)] = pad_temp_shared[(((rc_inner_outer * 48) + 8))];
        pad_temp_shared_local[(9)] = pad_temp_shared[(((rc_inner_outer * 48) + 9))];
        pad_temp_shared_local[(17)] = pad_temp_shared[(((rc_inner_outer * 48) + 10))];
        pad_temp_shared_local[(25)] = pad_temp_shared[(((rc_inner_outer * 48) + 11))];
        pad_temp_shared_local[(2)] = pad_temp_shared[(((rc_inner_outer * 48) + 14))];
        pad_temp_shared_local[(10)] = pad_temp_shared[(((rc_inner_outer * 48) + 15))];
        pad_temp_shared_local[(18)] = pad_temp_shared[(((rc_inner_outer * 48) + 16))];
        pad_temp_shared_local[(26)] = pad_temp_shared[(((rc_inner_outer * 48) + 17))];
        pad_temp_shared_local[(3)] = pad_temp_shared[(((rc_inner_outer * 48) + 20))];
        pad_temp_shared_local[(11)] = pad_temp_shared[(((rc_inner_outer * 48) + 21))];
        pad_temp_shared_local[(19)] = pad_temp_shared[(((rc_inner_outer * 48) + 22))];
        pad_temp_shared_local[(27)] = pad_temp_shared[(((rc_inner_outer * 48) + 23))];
        pad_temp_shared_local[(4)] = pad_temp_shared[(((rc_inner_outer * 48) + 26))];
        pad_temp_shared_local[(12)] = pad_temp_shared[(((rc_inner_outer * 48) + 27))];
        pad_temp_shared_local[(20)] = pad_temp_shared[(((rc_inner_outer * 48) + 28))];
        pad_temp_shared_local[(28)] = pad_temp_shared[(((rc_inner_outer * 48) + 29))];
        pad_temp_shared_local[(5)] = pad_temp_shared[(((rc_inner_outer * 48) + 32))];
        pad_temp_shared_local[(13)] = pad_temp_shared[(((rc_inner_outer * 48) + 33))];
        pad_temp_shared_local[(21)] = pad_temp_shared[(((rc_inner_outer * 48) + 34))];
        pad_temp_shared_local[(29)] = pad_temp_shared[(((rc_inner_outer * 48) + 35))];
        pad_temp_shared_local[(6)] = pad_temp_shared[(((rc_inner_outer * 48) + 38))];
        pad_temp_shared_local[(14)] = pad_temp_shared[(((rc_inner_outer * 48) + 39))];
        pad_temp_shared_local[(22)] = pad_temp_shared[(((rc_inner_outer * 48) + 40))];
        pad_temp_shared_local[(30)] = pad_temp_shared[(((rc_inner_outer * 48) + 41))];
        pad_temp_shared_local[(7)] = pad_temp_shared[(((rc_inner_outer * 48) + 44))];
        pad_temp_shared_local[(15)] = pad_temp_shared[(((rc_inner_outer * 48) + 45))];
        pad_temp_shared_local[(23)] = pad_temp_shared[(((rc_inner_outer * 48) + 46))];
        pad_temp_shared_local[(31)] = pad_temp_shared[(((rc_inner_outer * 48) + 47))];
        kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 2))];
        kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 386))];
        kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 5))];
        kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 48) + (rc_inner_outer * 6)) + 389))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(2)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(0)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(0)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(2)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(0)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(2)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(0)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(2)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(2)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(0)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(2)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(0)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(2)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(0)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(0)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(2)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(0)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(2)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(0)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(2)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(2)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(0)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(0)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(2)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(0)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(2)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
        compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(3)]));
        compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(1)]));
        compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
        compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(1)]));
        compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(3)]));
        compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(1)]));
        compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(3)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(1)]));
        compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(3)]));
        compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(1)]));
        compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(3)]));
        compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(1)]));
        compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
        compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(1)]));
        compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(3)]));
        compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(1)]));
        compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(3)]));
        compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(1)]));
        compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(3)]));
        compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(1)]));
        compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(3)]));
        compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(1)]));
        compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(3)]));
        compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
        compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(3)]));
        compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(1)]));
        compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
        compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(1)]));
        compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(3)]));
        compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(1)]));
        compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(3)]));
      }
    }
  }
  compute[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25088))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 1))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25089))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 2))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25090))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 3))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25091))] = compute_local[(28)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 56))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25144))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 57))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25145))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 58))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25146))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 59))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25147))] = compute_local[(29)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 112))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25200))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 113))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25201))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 114))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25202))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 115))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25203))] = compute_local[(30)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 168))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25256))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 169))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25257))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 170))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25258))] = compute_local[(27)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 171))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 224)) + (((int)blockIdx.x) * 4)) + 25259))] = compute_local[(31)];
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

    dim3 grid(14,14,2);
    dim3 block(1,1,8);

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


