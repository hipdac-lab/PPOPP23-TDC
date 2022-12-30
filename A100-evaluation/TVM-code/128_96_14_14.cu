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

#define C 128
#define N 96
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
  float compute_local[2];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[384];
  float pad_temp_shared_local[96];
  float kernel_shared_local[192];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[((((int)threadIdx.x) * 74))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((rc_outer * 12544) + (((((int)threadIdx.x) * 74) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 1))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 1) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 2))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 2) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 3))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 3) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 4))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 4) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 5))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 5) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 6))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 6) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 7))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 7) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 8))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 8) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 9))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 9) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 10))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 10) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 11))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 11) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 12))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 12) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 13))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 13) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 14))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 14) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 15))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 15) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 16))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((rc_outer * 12544) + (((((int)threadIdx.x) * 74) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 181))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 17))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 17) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 18))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 18) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 19))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 19) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 20))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 20) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 21))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 21) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 22))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 22) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 23))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 23) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 24))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 24) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 25))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 25) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 26))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 26) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 27))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 27) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 28))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 28) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 29))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 29) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 30))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 30) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 31))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 31) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 32))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((rc_outer * 12544) + (((((int)threadIdx.x) * 74) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 377))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 33))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 33) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 34))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 34) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 35))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 35) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 36))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 36) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 37))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 37) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 38))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 38) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 39))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 39) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 40))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 40) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 41))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 41) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 42))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 42) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 43))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 43) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 44))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 44) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 45))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 45) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 46))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 46) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 47))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 47) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 48))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((rc_outer * 12544) + (((((int)threadIdx.x) * 74) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 573))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 49))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 49) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 50))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 50) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 51))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 51) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 52))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 52) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 53))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 53) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 54))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 54) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 55))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 55) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 56))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 56) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 57))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 57) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 58))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 10) & 15))) && ((((((int)threadIdx.x) * 74) + 10) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 58) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 10) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 59))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 11) & 15))) && ((((((int)threadIdx.x) * 74) + 11) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 59) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 11) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 60))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 12) & 15))) && ((((((int)threadIdx.x) * 74) + 12) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 60) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 12) & 15)) - 15))] : 0.000000e+00f);
      pad_temp_shared[(((((int)threadIdx.x) * 74) + 61))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 13) & 15))) && ((((((int)threadIdx.x) * 74) + 13) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 61) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 13) & 15)) - 15))] : 0.000000e+00f);
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 62))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 14) & 15))) && ((((((int)threadIdx.x) * 74) + 14) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 62) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 14) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 63))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 15) & 15))) && ((((((int)threadIdx.x) * 74) + 15) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 63) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 15) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 64))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= ((((int)threadIdx.x) * 74) & 15))) && (((((int)threadIdx.x) * 74) & 15) < 15)) ? data[(((((((rc_outer * 12544) + (((((int)threadIdx.x) * 74) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 74) & 15)) + 769))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 65))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 1) & 15))) && ((((((int)threadIdx.x) * 74) + 1) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 65) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 1) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 66))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 2) & 15))) && ((((((int)threadIdx.x) * 74) + 2) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 66) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 2) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 67))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 3) & 15))) && ((((((int)threadIdx.x) * 74) + 3) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 67) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 3) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 68))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 4) & 15))) && ((((((int)threadIdx.x) * 74) + 4) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 68) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 4) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 69))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 5) & 15))) && ((((((int)threadIdx.x) * 74) + 5) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 69) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 5) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 70))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 6) & 15))) && ((((((int)threadIdx.x) * 74) + 6) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 70) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 6) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 71))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 7) & 15))) && ((((((int)threadIdx.x) * 74) + 7) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 71) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 7) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 72))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 8) & 15))) && ((((((int)threadIdx.x) * 74) + 8) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 72) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 8) & 15)) - 15))] : 0.000000e+00f);
      }
      if (((int)threadIdx.x) < 13) {
        pad_temp_shared[(((((int)threadIdx.x) * 74) + 73))] = (((((1 <= (((int)blockIdx.y) + ry_outer)) && ((((int)blockIdx.y) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 74) + 9) & 15))) && ((((((int)threadIdx.x) * 74) + 9) & 15) < 15)) ? data[(((((((rc_outer * 12544) + ((((((int)threadIdx.x) * 74) + 73) >> 4) * 196)) + (((int)blockIdx.y) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 74) + 9) & 15)) - 15))] : 0.000000e+00f);
      }
      kernel_shared[((((int)threadIdx.x) * 28))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((((int)threadIdx.x) * 28) / 192) * 1152)) + (rc_outer * 576)) + ((((((int)threadIdx.x) * 28) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 1))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 1) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 1) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 2))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 2) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 2) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 3))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 3) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 3) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 4))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 4) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 4) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 5))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 5) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 5) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 6))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 6) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 6) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 7))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 7) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 7) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 8))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 8) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 8) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 9))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 9) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 9) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 10))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 10) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 10) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 11))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 11) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 11) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 12))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 12) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 12) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 13))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 13) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 13) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 14))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 14) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 14) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 15))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 15) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 15) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 16))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 16) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 16) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 17))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 17) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 17) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 18))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 18) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 18) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      kernel_shared[(((((int)threadIdx.x) * 28) + 19))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 19) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 19) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 20))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 20) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 20) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 21))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 21) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 21) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 22))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 22) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 22) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 23))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 23) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 23) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 24))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 24) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 24) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 25))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 25) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 25) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 1) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 26))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 26) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 26) % 192) / 3) * 9)) + (ry_outer * 3)) + (((((int)threadIdx.x) * 28) + 2) % 3)))];
      }
      if (((int)threadIdx.x) < 13) {
        kernel_shared[(((((int)threadIdx.x) * 28) + 27))] = kernel[(((((((((int)blockIdx.z) * 2304) + ((((((int)threadIdx.x) * 28) + 27) / 192) * 1152)) + (rc_outer * 576)) + (((((((int)threadIdx.x) * 28) + 27) % 192) / 3) * 9)) + (ry_outer * 3)) + ((((int)threadIdx.x) * 28) % 3)))];
      }
      __syncthreads();
      for (int rc_inner_outer = 0; rc_inner_outer < 2; ++rc_inner_outer) {
        pad_temp_shared_local[(0)] = pad_temp_shared[(((rc_inner_outer * 512) + ((int)threadIdx.x)))];
        pad_temp_shared_local[(1)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 1))];
        pad_temp_shared_local[(2)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 2))];
        pad_temp_shared_local[(3)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 16))];
        pad_temp_shared_local[(4)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 17))];
        pad_temp_shared_local[(5)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 18))];
        pad_temp_shared_local[(6)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 32))];
        pad_temp_shared_local[(7)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 33))];
        pad_temp_shared_local[(8)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 34))];
        pad_temp_shared_local[(9)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 48))];
        pad_temp_shared_local[(10)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 49))];
        pad_temp_shared_local[(11)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 50))];
        pad_temp_shared_local[(12)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 64))];
        pad_temp_shared_local[(13)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 65))];
        pad_temp_shared_local[(14)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 66))];
        pad_temp_shared_local[(15)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 80))];
        pad_temp_shared_local[(16)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 81))];
        pad_temp_shared_local[(17)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 82))];
        pad_temp_shared_local[(18)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 96))];
        pad_temp_shared_local[(19)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 97))];
        pad_temp_shared_local[(20)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 98))];
        pad_temp_shared_local[(21)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 112))];
        pad_temp_shared_local[(22)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 113))];
        pad_temp_shared_local[(23)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 114))];
        pad_temp_shared_local[(24)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 128))];
        pad_temp_shared_local[(25)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 129))];
        pad_temp_shared_local[(26)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 130))];
        pad_temp_shared_local[(27)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 144))];
        pad_temp_shared_local[(28)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 145))];
        pad_temp_shared_local[(29)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 146))];
        pad_temp_shared_local[(30)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 160))];
        pad_temp_shared_local[(31)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 161))];
        pad_temp_shared_local[(32)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 162))];
        pad_temp_shared_local[(33)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 176))];
        pad_temp_shared_local[(34)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 177))];
        pad_temp_shared_local[(35)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 178))];
        pad_temp_shared_local[(36)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 192))];
        pad_temp_shared_local[(37)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 193))];
        pad_temp_shared_local[(38)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 194))];
        pad_temp_shared_local[(39)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 208))];
        pad_temp_shared_local[(40)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 209))];
        pad_temp_shared_local[(41)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 210))];
        pad_temp_shared_local[(42)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 224))];
        pad_temp_shared_local[(43)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 225))];
        pad_temp_shared_local[(44)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 226))];
        pad_temp_shared_local[(45)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 240))];
        pad_temp_shared_local[(46)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 241))];
        pad_temp_shared_local[(47)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 242))];
        pad_temp_shared_local[(48)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 256))];
        pad_temp_shared_local[(49)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 257))];
        pad_temp_shared_local[(50)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 258))];
        pad_temp_shared_local[(51)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 272))];
        pad_temp_shared_local[(52)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 273))];
        pad_temp_shared_local[(53)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 274))];
        pad_temp_shared_local[(54)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 288))];
        pad_temp_shared_local[(55)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 289))];
        pad_temp_shared_local[(56)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 290))];
        pad_temp_shared_local[(57)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 304))];
        pad_temp_shared_local[(58)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 305))];
        pad_temp_shared_local[(59)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 306))];
        pad_temp_shared_local[(60)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 320))];
        pad_temp_shared_local[(61)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 321))];
        pad_temp_shared_local[(62)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 322))];
        pad_temp_shared_local[(63)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 336))];
        pad_temp_shared_local[(64)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 337))];
        pad_temp_shared_local[(65)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 338))];
        pad_temp_shared_local[(66)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 352))];
        pad_temp_shared_local[(67)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 353))];
        pad_temp_shared_local[(68)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 354))];
        pad_temp_shared_local[(69)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 368))];
        pad_temp_shared_local[(70)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 369))];
        pad_temp_shared_local[(71)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 370))];
        pad_temp_shared_local[(72)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 384))];
        pad_temp_shared_local[(73)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 385))];
        pad_temp_shared_local[(74)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 386))];
        pad_temp_shared_local[(75)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 400))];
        pad_temp_shared_local[(76)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 401))];
        pad_temp_shared_local[(77)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 402))];
        pad_temp_shared_local[(78)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 416))];
        pad_temp_shared_local[(79)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 417))];
        pad_temp_shared_local[(80)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 418))];
        pad_temp_shared_local[(81)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 432))];
        pad_temp_shared_local[(82)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 433))];
        pad_temp_shared_local[(83)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 434))];
        pad_temp_shared_local[(84)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 448))];
        pad_temp_shared_local[(85)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 449))];
        pad_temp_shared_local[(86)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 450))];
        pad_temp_shared_local[(87)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 464))];
        pad_temp_shared_local[(88)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 465))];
        pad_temp_shared_local[(89)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 466))];
        pad_temp_shared_local[(90)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 480))];
        pad_temp_shared_local[(91)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 481))];
        pad_temp_shared_local[(92)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 482))];
        pad_temp_shared_local[(93)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 496))];
        pad_temp_shared_local[(94)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 497))];
        pad_temp_shared_local[(95)] = pad_temp_shared[((((rc_inner_outer * 512) + ((int)threadIdx.x)) + 498))];
        kernel_shared_local[(0)] = kernel_shared[((rc_inner_outer * 96))];
        kernel_shared_local[(1)] = kernel_shared[(((rc_inner_outer * 96) + 1))];
        kernel_shared_local[(2)] = kernel_shared[(((rc_inner_outer * 96) + 2))];
        kernel_shared_local[(3)] = kernel_shared[(((rc_inner_outer * 96) + 3))];
        kernel_shared_local[(4)] = kernel_shared[(((rc_inner_outer * 96) + 4))];
        kernel_shared_local[(5)] = kernel_shared[(((rc_inner_outer * 96) + 5))];
        kernel_shared_local[(6)] = kernel_shared[(((rc_inner_outer * 96) + 6))];
        kernel_shared_local[(7)] = kernel_shared[(((rc_inner_outer * 96) + 7))];
        kernel_shared_local[(8)] = kernel_shared[(((rc_inner_outer * 96) + 8))];
        kernel_shared_local[(9)] = kernel_shared[(((rc_inner_outer * 96) + 9))];
        kernel_shared_local[(10)] = kernel_shared[(((rc_inner_outer * 96) + 10))];
        kernel_shared_local[(11)] = kernel_shared[(((rc_inner_outer * 96) + 11))];
        kernel_shared_local[(12)] = kernel_shared[(((rc_inner_outer * 96) + 12))];
        kernel_shared_local[(13)] = kernel_shared[(((rc_inner_outer * 96) + 13))];
        kernel_shared_local[(14)] = kernel_shared[(((rc_inner_outer * 96) + 14))];
        kernel_shared_local[(15)] = kernel_shared[(((rc_inner_outer * 96) + 15))];
        kernel_shared_local[(16)] = kernel_shared[(((rc_inner_outer * 96) + 16))];
        kernel_shared_local[(17)] = kernel_shared[(((rc_inner_outer * 96) + 17))];
        kernel_shared_local[(18)] = kernel_shared[(((rc_inner_outer * 96) + 18))];
        kernel_shared_local[(19)] = kernel_shared[(((rc_inner_outer * 96) + 19))];
        kernel_shared_local[(20)] = kernel_shared[(((rc_inner_outer * 96) + 20))];
        kernel_shared_local[(21)] = kernel_shared[(((rc_inner_outer * 96) + 21))];
        kernel_shared_local[(22)] = kernel_shared[(((rc_inner_outer * 96) + 22))];
        kernel_shared_local[(23)] = kernel_shared[(((rc_inner_outer * 96) + 23))];
        kernel_shared_local[(24)] = kernel_shared[(((rc_inner_outer * 96) + 24))];
        kernel_shared_local[(25)] = kernel_shared[(((rc_inner_outer * 96) + 25))];
        kernel_shared_local[(26)] = kernel_shared[(((rc_inner_outer * 96) + 26))];
        kernel_shared_local[(27)] = kernel_shared[(((rc_inner_outer * 96) + 27))];
        kernel_shared_local[(28)] = kernel_shared[(((rc_inner_outer * 96) + 28))];
        kernel_shared_local[(29)] = kernel_shared[(((rc_inner_outer * 96) + 29))];
        kernel_shared_local[(30)] = kernel_shared[(((rc_inner_outer * 96) + 30))];
        kernel_shared_local[(31)] = kernel_shared[(((rc_inner_outer * 96) + 31))];
        kernel_shared_local[(32)] = kernel_shared[(((rc_inner_outer * 96) + 32))];
        kernel_shared_local[(33)] = kernel_shared[(((rc_inner_outer * 96) + 33))];
        kernel_shared_local[(34)] = kernel_shared[(((rc_inner_outer * 96) + 34))];
        kernel_shared_local[(35)] = kernel_shared[(((rc_inner_outer * 96) + 35))];
        kernel_shared_local[(36)] = kernel_shared[(((rc_inner_outer * 96) + 36))];
        kernel_shared_local[(37)] = kernel_shared[(((rc_inner_outer * 96) + 37))];
        kernel_shared_local[(38)] = kernel_shared[(((rc_inner_outer * 96) + 38))];
        kernel_shared_local[(39)] = kernel_shared[(((rc_inner_outer * 96) + 39))];
        kernel_shared_local[(40)] = kernel_shared[(((rc_inner_outer * 96) + 40))];
        kernel_shared_local[(41)] = kernel_shared[(((rc_inner_outer * 96) + 41))];
        kernel_shared_local[(42)] = kernel_shared[(((rc_inner_outer * 96) + 42))];
        kernel_shared_local[(43)] = kernel_shared[(((rc_inner_outer * 96) + 43))];
        kernel_shared_local[(44)] = kernel_shared[(((rc_inner_outer * 96) + 44))];
        kernel_shared_local[(45)] = kernel_shared[(((rc_inner_outer * 96) + 45))];
        kernel_shared_local[(46)] = kernel_shared[(((rc_inner_outer * 96) + 46))];
        kernel_shared_local[(47)] = kernel_shared[(((rc_inner_outer * 96) + 47))];
        kernel_shared_local[(48)] = kernel_shared[(((rc_inner_outer * 96) + 48))];
        kernel_shared_local[(49)] = kernel_shared[(((rc_inner_outer * 96) + 49))];
        kernel_shared_local[(50)] = kernel_shared[(((rc_inner_outer * 96) + 50))];
        kernel_shared_local[(51)] = kernel_shared[(((rc_inner_outer * 96) + 51))];
        kernel_shared_local[(52)] = kernel_shared[(((rc_inner_outer * 96) + 52))];
        kernel_shared_local[(53)] = kernel_shared[(((rc_inner_outer * 96) + 53))];
        kernel_shared_local[(54)] = kernel_shared[(((rc_inner_outer * 96) + 54))];
        kernel_shared_local[(55)] = kernel_shared[(((rc_inner_outer * 96) + 55))];
        kernel_shared_local[(56)] = kernel_shared[(((rc_inner_outer * 96) + 56))];
        kernel_shared_local[(57)] = kernel_shared[(((rc_inner_outer * 96) + 57))];
        kernel_shared_local[(58)] = kernel_shared[(((rc_inner_outer * 96) + 58))];
        kernel_shared_local[(59)] = kernel_shared[(((rc_inner_outer * 96) + 59))];
        kernel_shared_local[(60)] = kernel_shared[(((rc_inner_outer * 96) + 60))];
        kernel_shared_local[(61)] = kernel_shared[(((rc_inner_outer * 96) + 61))];
        kernel_shared_local[(62)] = kernel_shared[(((rc_inner_outer * 96) + 62))];
        kernel_shared_local[(63)] = kernel_shared[(((rc_inner_outer * 96) + 63))];
        kernel_shared_local[(64)] = kernel_shared[(((rc_inner_outer * 96) + 64))];
        kernel_shared_local[(65)] = kernel_shared[(((rc_inner_outer * 96) + 65))];
        kernel_shared_local[(66)] = kernel_shared[(((rc_inner_outer * 96) + 66))];
        kernel_shared_local[(67)] = kernel_shared[(((rc_inner_outer * 96) + 67))];
        kernel_shared_local[(68)] = kernel_shared[(((rc_inner_outer * 96) + 68))];
        kernel_shared_local[(69)] = kernel_shared[(((rc_inner_outer * 96) + 69))];
        kernel_shared_local[(70)] = kernel_shared[(((rc_inner_outer * 96) + 70))];
        kernel_shared_local[(71)] = kernel_shared[(((rc_inner_outer * 96) + 71))];
        kernel_shared_local[(72)] = kernel_shared[(((rc_inner_outer * 96) + 72))];
        kernel_shared_local[(73)] = kernel_shared[(((rc_inner_outer * 96) + 73))];
        kernel_shared_local[(74)] = kernel_shared[(((rc_inner_outer * 96) + 74))];
        kernel_shared_local[(75)] = kernel_shared[(((rc_inner_outer * 96) + 75))];
        kernel_shared_local[(76)] = kernel_shared[(((rc_inner_outer * 96) + 76))];
        kernel_shared_local[(77)] = kernel_shared[(((rc_inner_outer * 96) + 77))];
        kernel_shared_local[(78)] = kernel_shared[(((rc_inner_outer * 96) + 78))];
        kernel_shared_local[(79)] = kernel_shared[(((rc_inner_outer * 96) + 79))];
        kernel_shared_local[(80)] = kernel_shared[(((rc_inner_outer * 96) + 80))];
        kernel_shared_local[(81)] = kernel_shared[(((rc_inner_outer * 96) + 81))];
        kernel_shared_local[(82)] = kernel_shared[(((rc_inner_outer * 96) + 82))];
        kernel_shared_local[(83)] = kernel_shared[(((rc_inner_outer * 96) + 83))];
        kernel_shared_local[(84)] = kernel_shared[(((rc_inner_outer * 96) + 84))];
        kernel_shared_local[(85)] = kernel_shared[(((rc_inner_outer * 96) + 85))];
        kernel_shared_local[(86)] = kernel_shared[(((rc_inner_outer * 96) + 86))];
        kernel_shared_local[(87)] = kernel_shared[(((rc_inner_outer * 96) + 87))];
        kernel_shared_local[(88)] = kernel_shared[(((rc_inner_outer * 96) + 88))];
        kernel_shared_local[(89)] = kernel_shared[(((rc_inner_outer * 96) + 89))];
        kernel_shared_local[(90)] = kernel_shared[(((rc_inner_outer * 96) + 90))];
        kernel_shared_local[(91)] = kernel_shared[(((rc_inner_outer * 96) + 91))];
        kernel_shared_local[(92)] = kernel_shared[(((rc_inner_outer * 96) + 92))];
        kernel_shared_local[(93)] = kernel_shared[(((rc_inner_outer * 96) + 93))];
        kernel_shared_local[(94)] = kernel_shared[(((rc_inner_outer * 96) + 94))];
        kernel_shared_local[(95)] = kernel_shared[(((rc_inner_outer * 96) + 95))];
        kernel_shared_local[(96)] = kernel_shared[(((rc_inner_outer * 96) + 192))];
        kernel_shared_local[(97)] = kernel_shared[(((rc_inner_outer * 96) + 193))];
        kernel_shared_local[(98)] = kernel_shared[(((rc_inner_outer * 96) + 194))];
        kernel_shared_local[(99)] = kernel_shared[(((rc_inner_outer * 96) + 195))];
        kernel_shared_local[(100)] = kernel_shared[(((rc_inner_outer * 96) + 196))];
        kernel_shared_local[(101)] = kernel_shared[(((rc_inner_outer * 96) + 197))];
        kernel_shared_local[(102)] = kernel_shared[(((rc_inner_outer * 96) + 198))];
        kernel_shared_local[(103)] = kernel_shared[(((rc_inner_outer * 96) + 199))];
        kernel_shared_local[(104)] = kernel_shared[(((rc_inner_outer * 96) + 200))];
        kernel_shared_local[(105)] = kernel_shared[(((rc_inner_outer * 96) + 201))];
        kernel_shared_local[(106)] = kernel_shared[(((rc_inner_outer * 96) + 202))];
        kernel_shared_local[(107)] = kernel_shared[(((rc_inner_outer * 96) + 203))];
        kernel_shared_local[(108)] = kernel_shared[(((rc_inner_outer * 96) + 204))];
        kernel_shared_local[(109)] = kernel_shared[(((rc_inner_outer * 96) + 205))];
        kernel_shared_local[(110)] = kernel_shared[(((rc_inner_outer * 96) + 206))];
        kernel_shared_local[(111)] = kernel_shared[(((rc_inner_outer * 96) + 207))];
        kernel_shared_local[(112)] = kernel_shared[(((rc_inner_outer * 96) + 208))];
        kernel_shared_local[(113)] = kernel_shared[(((rc_inner_outer * 96) + 209))];
        kernel_shared_local[(114)] = kernel_shared[(((rc_inner_outer * 96) + 210))];
        kernel_shared_local[(115)] = kernel_shared[(((rc_inner_outer * 96) + 211))];
        kernel_shared_local[(116)] = kernel_shared[(((rc_inner_outer * 96) + 212))];
        kernel_shared_local[(117)] = kernel_shared[(((rc_inner_outer * 96) + 213))];
        kernel_shared_local[(118)] = kernel_shared[(((rc_inner_outer * 96) + 214))];
        kernel_shared_local[(119)] = kernel_shared[(((rc_inner_outer * 96) + 215))];
        kernel_shared_local[(120)] = kernel_shared[(((rc_inner_outer * 96) + 216))];
        kernel_shared_local[(121)] = kernel_shared[(((rc_inner_outer * 96) + 217))];
        kernel_shared_local[(122)] = kernel_shared[(((rc_inner_outer * 96) + 218))];
        kernel_shared_local[(123)] = kernel_shared[(((rc_inner_outer * 96) + 219))];
        kernel_shared_local[(124)] = kernel_shared[(((rc_inner_outer * 96) + 220))];
        kernel_shared_local[(125)] = kernel_shared[(((rc_inner_outer * 96) + 221))];
        kernel_shared_local[(126)] = kernel_shared[(((rc_inner_outer * 96) + 222))];
        kernel_shared_local[(127)] = kernel_shared[(((rc_inner_outer * 96) + 223))];
        kernel_shared_local[(128)] = kernel_shared[(((rc_inner_outer * 96) + 224))];
        kernel_shared_local[(129)] = kernel_shared[(((rc_inner_outer * 96) + 225))];
        kernel_shared_local[(130)] = kernel_shared[(((rc_inner_outer * 96) + 226))];
        kernel_shared_local[(131)] = kernel_shared[(((rc_inner_outer * 96) + 227))];
        kernel_shared_local[(132)] = kernel_shared[(((rc_inner_outer * 96) + 228))];
        kernel_shared_local[(133)] = kernel_shared[(((rc_inner_outer * 96) + 229))];
        kernel_shared_local[(134)] = kernel_shared[(((rc_inner_outer * 96) + 230))];
        kernel_shared_local[(135)] = kernel_shared[(((rc_inner_outer * 96) + 231))];
        kernel_shared_local[(136)] = kernel_shared[(((rc_inner_outer * 96) + 232))];
        kernel_shared_local[(137)] = kernel_shared[(((rc_inner_outer * 96) + 233))];
        kernel_shared_local[(138)] = kernel_shared[(((rc_inner_outer * 96) + 234))];
        kernel_shared_local[(139)] = kernel_shared[(((rc_inner_outer * 96) + 235))];
        kernel_shared_local[(140)] = kernel_shared[(((rc_inner_outer * 96) + 236))];
        kernel_shared_local[(141)] = kernel_shared[(((rc_inner_outer * 96) + 237))];
        kernel_shared_local[(142)] = kernel_shared[(((rc_inner_outer * 96) + 238))];
        kernel_shared_local[(143)] = kernel_shared[(((rc_inner_outer * 96) + 239))];
        kernel_shared_local[(144)] = kernel_shared[(((rc_inner_outer * 96) + 240))];
        kernel_shared_local[(145)] = kernel_shared[(((rc_inner_outer * 96) + 241))];
        kernel_shared_local[(146)] = kernel_shared[(((rc_inner_outer * 96) + 242))];
        kernel_shared_local[(147)] = kernel_shared[(((rc_inner_outer * 96) + 243))];
        kernel_shared_local[(148)] = kernel_shared[(((rc_inner_outer * 96) + 244))];
        kernel_shared_local[(149)] = kernel_shared[(((rc_inner_outer * 96) + 245))];
        kernel_shared_local[(150)] = kernel_shared[(((rc_inner_outer * 96) + 246))];
        kernel_shared_local[(151)] = kernel_shared[(((rc_inner_outer * 96) + 247))];
        kernel_shared_local[(152)] = kernel_shared[(((rc_inner_outer * 96) + 248))];
        kernel_shared_local[(153)] = kernel_shared[(((rc_inner_outer * 96) + 249))];
        kernel_shared_local[(154)] = kernel_shared[(((rc_inner_outer * 96) + 250))];
        kernel_shared_local[(155)] = kernel_shared[(((rc_inner_outer * 96) + 251))];
        kernel_shared_local[(156)] = kernel_shared[(((rc_inner_outer * 96) + 252))];
        kernel_shared_local[(157)] = kernel_shared[(((rc_inner_outer * 96) + 253))];
        kernel_shared_local[(158)] = kernel_shared[(((rc_inner_outer * 96) + 254))];
        kernel_shared_local[(159)] = kernel_shared[(((rc_inner_outer * 96) + 255))];
        kernel_shared_local[(160)] = kernel_shared[(((rc_inner_outer * 96) + 256))];
        kernel_shared_local[(161)] = kernel_shared[(((rc_inner_outer * 96) + 257))];
        kernel_shared_local[(162)] = kernel_shared[(((rc_inner_outer * 96) + 258))];
        kernel_shared_local[(163)] = kernel_shared[(((rc_inner_outer * 96) + 259))];
        kernel_shared_local[(164)] = kernel_shared[(((rc_inner_outer * 96) + 260))];
        kernel_shared_local[(165)] = kernel_shared[(((rc_inner_outer * 96) + 261))];
        kernel_shared_local[(166)] = kernel_shared[(((rc_inner_outer * 96) + 262))];
        kernel_shared_local[(167)] = kernel_shared[(((rc_inner_outer * 96) + 263))];
        kernel_shared_local[(168)] = kernel_shared[(((rc_inner_outer * 96) + 264))];
        kernel_shared_local[(169)] = kernel_shared[(((rc_inner_outer * 96) + 265))];
        kernel_shared_local[(170)] = kernel_shared[(((rc_inner_outer * 96) + 266))];
        kernel_shared_local[(171)] = kernel_shared[(((rc_inner_outer * 96) + 267))];
        kernel_shared_local[(172)] = kernel_shared[(((rc_inner_outer * 96) + 268))];
        kernel_shared_local[(173)] = kernel_shared[(((rc_inner_outer * 96) + 269))];
        kernel_shared_local[(174)] = kernel_shared[(((rc_inner_outer * 96) + 270))];
        kernel_shared_local[(175)] = kernel_shared[(((rc_inner_outer * 96) + 271))];
        kernel_shared_local[(176)] = kernel_shared[(((rc_inner_outer * 96) + 272))];
        kernel_shared_local[(177)] = kernel_shared[(((rc_inner_outer * 96) + 273))];
        kernel_shared_local[(178)] = kernel_shared[(((rc_inner_outer * 96) + 274))];
        kernel_shared_local[(179)] = kernel_shared[(((rc_inner_outer * 96) + 275))];
        kernel_shared_local[(180)] = kernel_shared[(((rc_inner_outer * 96) + 276))];
        kernel_shared_local[(181)] = kernel_shared[(((rc_inner_outer * 96) + 277))];
        kernel_shared_local[(182)] = kernel_shared[(((rc_inner_outer * 96) + 278))];
        kernel_shared_local[(183)] = kernel_shared[(((rc_inner_outer * 96) + 279))];
        kernel_shared_local[(184)] = kernel_shared[(((rc_inner_outer * 96) + 280))];
        kernel_shared_local[(185)] = kernel_shared[(((rc_inner_outer * 96) + 281))];
        kernel_shared_local[(186)] = kernel_shared[(((rc_inner_outer * 96) + 282))];
        kernel_shared_local[(187)] = kernel_shared[(((rc_inner_outer * 96) + 283))];
        kernel_shared_local[(188)] = kernel_shared[(((rc_inner_outer * 96) + 284))];
        kernel_shared_local[(189)] = kernel_shared[(((rc_inner_outer * 96) + 285))];
        kernel_shared_local[(190)] = kernel_shared[(((rc_inner_outer * 96) + 286))];
        kernel_shared_local[(191)] = kernel_shared[(((rc_inner_outer * 96) + 287))];
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(96)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(97)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(98)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(99)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(100)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(101)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(6)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(102)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(7)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(103)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(8)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(104)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(9)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(105)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(10)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(106)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(11)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(107)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(12)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(108)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(13)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(109)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(14)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(110)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(15)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(111)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(16)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(112)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(17)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(113)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(18)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(114)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(19)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(115)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(20)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(116)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(21)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(117)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(22)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(118)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(23)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(119)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(24)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(120)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(25)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(121)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(26)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(122)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(27)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(123)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(28)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(124)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(29)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(125)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(30)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(126)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(31)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(127)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(32)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(128)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(33)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(129)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(34)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(130)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(35)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(131)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(36)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(132)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(37)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(133)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(38)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(134)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(39)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(135)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(40)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(136)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(41)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(137)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(42)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(138)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(43)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(139)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(44)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(140)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(45)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(141)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(46)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(142)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(47)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(143)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(48)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(48)] * kernel_shared_local[(144)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(49)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(49)] * kernel_shared_local[(145)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(50)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(50)] * kernel_shared_local[(146)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(51)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(51)] * kernel_shared_local[(147)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(52)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(52)] * kernel_shared_local[(148)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(53)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(53)] * kernel_shared_local[(149)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(54)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(54)] * kernel_shared_local[(150)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(55)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(55)] * kernel_shared_local[(151)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(56)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(56)] * kernel_shared_local[(152)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(57)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(57)] * kernel_shared_local[(153)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(58)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(58)] * kernel_shared_local[(154)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(59)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(59)] * kernel_shared_local[(155)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(60)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(60)] * kernel_shared_local[(156)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(61)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(61)] * kernel_shared_local[(157)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(62)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(62)] * kernel_shared_local[(158)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(63)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(63)] * kernel_shared_local[(159)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(64)] * kernel_shared_local[(64)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(64)] * kernel_shared_local[(160)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(65)] * kernel_shared_local[(65)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(65)] * kernel_shared_local[(161)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(66)] * kernel_shared_local[(66)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(66)] * kernel_shared_local[(162)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(67)] * kernel_shared_local[(67)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(67)] * kernel_shared_local[(163)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(68)] * kernel_shared_local[(68)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(68)] * kernel_shared_local[(164)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(69)] * kernel_shared_local[(69)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(69)] * kernel_shared_local[(165)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(70)] * kernel_shared_local[(70)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(70)] * kernel_shared_local[(166)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(71)] * kernel_shared_local[(71)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(71)] * kernel_shared_local[(167)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(72)] * kernel_shared_local[(72)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(72)] * kernel_shared_local[(168)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(73)] * kernel_shared_local[(73)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(73)] * kernel_shared_local[(169)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(74)] * kernel_shared_local[(74)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(74)] * kernel_shared_local[(170)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(75)] * kernel_shared_local[(75)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(75)] * kernel_shared_local[(171)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(76)] * kernel_shared_local[(76)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(76)] * kernel_shared_local[(172)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(77)] * kernel_shared_local[(77)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(77)] * kernel_shared_local[(173)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(78)] * kernel_shared_local[(78)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(78)] * kernel_shared_local[(174)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(79)] * kernel_shared_local[(79)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(79)] * kernel_shared_local[(175)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(80)] * kernel_shared_local[(80)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(80)] * kernel_shared_local[(176)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(81)] * kernel_shared_local[(81)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(81)] * kernel_shared_local[(177)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(82)] * kernel_shared_local[(82)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(82)] * kernel_shared_local[(178)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(83)] * kernel_shared_local[(83)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(83)] * kernel_shared_local[(179)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(84)] * kernel_shared_local[(84)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(84)] * kernel_shared_local[(180)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(85)] * kernel_shared_local[(85)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(85)] * kernel_shared_local[(181)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(86)] * kernel_shared_local[(86)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(86)] * kernel_shared_local[(182)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(87)] * kernel_shared_local[(87)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(87)] * kernel_shared_local[(183)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(88)] * kernel_shared_local[(88)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(88)] * kernel_shared_local[(184)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(89)] * kernel_shared_local[(89)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(89)] * kernel_shared_local[(185)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(90)] * kernel_shared_local[(90)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(90)] * kernel_shared_local[(186)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(91)] * kernel_shared_local[(91)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(91)] * kernel_shared_local[(187)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(92)] * kernel_shared_local[(92)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(92)] * kernel_shared_local[(188)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(93)] * kernel_shared_local[(93)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(93)] * kernel_shared_local[(189)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(94)] * kernel_shared_local[(94)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(94)] * kernel_shared_local[(190)]));
        compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(95)] * kernel_shared_local[(95)]));
        compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(95)] * kernel_shared_local[(191)]));
      }
    }
  }
  compute[((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 14)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.z) * 392) + (((int)blockIdx.y) * 14)) + ((int)threadIdx.x)) + 196))] = compute_local[(1)];
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

    dim3 grid(1,14,48);
    dim3 block(14,1,1);

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


