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

#define C 32
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
  float compute_local[4];
  __shared__ float pad_temp_shared[9600];
  __shared__ float kernel_shared[2304];
  float pad_temp_shared_local[48];
  float kernel_shared_local[12];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  pad_temp_shared[((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)))] = (((((1 <= ((((int)blockIdx.y) * 28) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 300) / 10))) && (((((int)blockIdx.y) * 28) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 1))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 2))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 3))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 4))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 5))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 6))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 7))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 8))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 9))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 10))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 10) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 10) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 10) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 10) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 11))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 11) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 11) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 11) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 11) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 12))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 12) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 12) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 12) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 12) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 2) % 10)) - 57))] : 0.000000e+00f);
  pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 13))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 13) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 13) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 13) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 13) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 3) % 10)) - 57))] : 0.000000e+00f);
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9586) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1186) {
          pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 14))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 14) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 4) % 10)) - 57))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9585) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1185) {
          pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 15))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 15) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 5) % 10)) - 57))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9584) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1184) {
          pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 16))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 16) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 6) % 10)) - 57))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9583) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1183) {
          pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 17))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 17) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 7) % 10)) - 57))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9582) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1182) {
          if (((int)threadIdx.x) < 7) {
            pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 18))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 18) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 8) % 10)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9581) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1181) {
          if (((int)threadIdx.x) < 7) {
            pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 19))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 19) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 9) % 10)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 20) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) / 10)) < 958) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9580) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1180) {
          if (((int)threadIdx.x) < 7) {
            pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 20))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 20) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 20) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)))) && (((((int)blockIdx.x) * 8) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 20) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 20) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) % 10)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 4) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) / 300)) < 32) {
    if (((((int)threadIdx.z) * 120) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) / 10)) < 960) {
      if ((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) < 9579) {
        if (((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) < 1179) {
          if (((int)threadIdx.x) < 7) {
            pad_temp_shared[(((((((int)threadIdx.z) * 1200) + (((int)threadIdx.y) * 172)) + (((int)threadIdx.x) * 22)) + 21))] = (((((1 <= ((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) % 300) / 10))) && (((((int)blockIdx.y) * 28) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) % 300) / 10)) < 57)) && (1 <= ((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)))) && (((((int)blockIdx.x) * 8) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) < 57)) ? data[((((((((((int)threadIdx.z) * 12544) + (((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) / 300) * 3136)) + (((int)blockIdx.y) * 1568)) + ((((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 21) % 300) / 10) * 56)) + (((int)blockIdx.x) * 8)) + ((((((int)threadIdx.y) * 172) + (((int)threadIdx.x) * 22)) + 1) % 10)) - 57))] : 0.000000e+00f);
          }
        }
      }
    }
  }
  if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 768) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2304) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 288) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)))] = kernel[(((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)))];
              }
            }
          }
        }
      }
    }
  }
  if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 768) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2303) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 287) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 1))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 1))];
              }
            }
          }
        }
      }
    }
  }
  if (((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 768) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2302) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 286) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + (((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 2))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 2))];
              }
            }
          }
        }
      }
    }
  }
  if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 767) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2301) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 285) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 3))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 3))];
              }
            }
          }
        }
      }
    }
  }
  if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 767) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2300) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 284) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 4))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 4))];
              }
            }
          }
        }
      }
    }
  }
  if ((((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96) + ((int)threadIdx.z)) < 8) {
    if (((((int)threadIdx.z) * 32) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 3)) < 256) {
      if ((((((int)threadIdx.z) * 96) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) < 767) {
        if ((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) < 2299) {
          if (((((int)threadIdx.y) * 42) + (((int)threadIdx.x) * 6)) < 283) {
            if (((int)threadIdx.x) < 7) {
              if ((((((int)blockIdx.z) * 8) + ((((((int)threadIdx.y) * 14) + (((int)threadIdx.x) * 2)) + 1) / 96)) + ((int)threadIdx.z)) < 32) {
                kernel_shared[(((((((int)threadIdx.z) * 288) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 5))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 42)) + (((int)threadIdx.x) * 6)) + 5))];
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  for (int rc_inner_outer = 0; rc_inner_outer < 8; ++rc_inner_outer) {
    pad_temp_shared_local[(0)] = pad_temp_shared[((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 12))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 30))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 31))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 32))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 300))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 301))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 302))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 310))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 311))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 312))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 320))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 321))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 330))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 331))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 332))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 600))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 601))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 602))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 610))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 611))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 612))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 620))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 621))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 622))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 631))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 632))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 900))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 901))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 902))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 910))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 911))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 912))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 920))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 921))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 922))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 930))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 931))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 932))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 1))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 2))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 9))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 10))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 11))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 18))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 19))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 20))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 27))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 28))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 29))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(4)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(5)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 12))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 30))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 31))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 32))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 40))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 41))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 42))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 310))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 311))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 312))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 320))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 321))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 330))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 331))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 332))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 340))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 341))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 342))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 610))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 611))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 612))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 620))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 621))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 622))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 631))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 632))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 640))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 641))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 642))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 910))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 911))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 912))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 920))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 921))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 922))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 930))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 931))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 932))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 940))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 941))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 942))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 3))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 4))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 5))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 12))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 13))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 14))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 21))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 22))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 23))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 30))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 31))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 32))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(4)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(5)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(1)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 21))];
    pad_temp_shared_local[(2)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 22))];
    pad_temp_shared_local[(3)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 30))];
    pad_temp_shared_local[(4)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 31))];
    pad_temp_shared_local[(5)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 32))];
    pad_temp_shared_local[(6)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 40))];
    pad_temp_shared_local[(7)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 41))];
    pad_temp_shared_local[(8)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 42))];
    pad_temp_shared_local[(9)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 50))];
    pad_temp_shared_local[(10)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 51))];
    pad_temp_shared_local[(11)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 52))];
    pad_temp_shared_local[(12)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 320))];
    pad_temp_shared_local[(13)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 321))];
    pad_temp_shared_local[(14)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 322))];
    pad_temp_shared_local[(15)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 330))];
    pad_temp_shared_local[(16)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 331))];
    pad_temp_shared_local[(17)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 332))];
    pad_temp_shared_local[(18)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 340))];
    pad_temp_shared_local[(19)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 341))];
    pad_temp_shared_local[(20)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 342))];
    pad_temp_shared_local[(21)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 350))];
    pad_temp_shared_local[(22)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 351))];
    pad_temp_shared_local[(23)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 352))];
    pad_temp_shared_local[(24)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 620))];
    pad_temp_shared_local[(25)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 621))];
    pad_temp_shared_local[(26)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 622))];
    pad_temp_shared_local[(27)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 630))];
    pad_temp_shared_local[(28)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 631))];
    pad_temp_shared_local[(29)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 632))];
    pad_temp_shared_local[(30)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 640))];
    pad_temp_shared_local[(31)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 641))];
    pad_temp_shared_local[(32)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 642))];
    pad_temp_shared_local[(33)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 650))];
    pad_temp_shared_local[(34)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 651))];
    pad_temp_shared_local[(35)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 652))];
    pad_temp_shared_local[(36)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 920))];
    pad_temp_shared_local[(37)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 921))];
    pad_temp_shared_local[(38)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 922))];
    pad_temp_shared_local[(39)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 930))];
    pad_temp_shared_local[(40)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 931))];
    pad_temp_shared_local[(41)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 932))];
    pad_temp_shared_local[(42)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 940))];
    pad_temp_shared_local[(43)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 941))];
    pad_temp_shared_local[(44)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 942))];
    pad_temp_shared_local[(45)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 950))];
    pad_temp_shared_local[(46)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 951))];
    pad_temp_shared_local[(47)] = pad_temp_shared[(((((rc_inner_outer * 1200) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)) + 952))];
    kernel_shared_local[(0)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 6))];
    kernel_shared_local[(1)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 7))];
    kernel_shared_local[(2)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 8))];
    kernel_shared_local[(3)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 15))];
    kernel_shared_local[(4)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 16))];
    kernel_shared_local[(5)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 17))];
    kernel_shared_local[(6)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 24))];
    kernel_shared_local[(7)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 25))];
    kernel_shared_local[(8)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 26))];
    kernel_shared_local[(9)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 33))];
    kernel_shared_local[(10)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 34))];
    kernel_shared_local[(11)] = kernel_shared[((((((int)threadIdx.z) * 288) + (rc_inner_outer * 36)) + 35))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(0)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(6)] * kernel_shared_local[(0)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(9)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(1)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(7)] * kernel_shared_local[(1)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(10)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(2)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(8)] * kernel_shared_local[(2)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(11)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(12)] * kernel_shared_local[(3)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(15)] * kernel_shared_local[(3)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(18)] * kernel_shared_local[(3)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(21)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(13)] * kernel_shared_local[(4)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(16)] * kernel_shared_local[(4)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(19)] * kernel_shared_local[(4)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(22)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(14)] * kernel_shared_local[(5)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(17)] * kernel_shared_local[(5)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(20)] * kernel_shared_local[(5)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(23)] * kernel_shared_local[(5)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(24)] * kernel_shared_local[(6)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(27)] * kernel_shared_local[(6)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(30)] * kernel_shared_local[(6)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(33)] * kernel_shared_local[(6)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(25)] * kernel_shared_local[(7)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(28)] * kernel_shared_local[(7)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(31)] * kernel_shared_local[(7)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(34)] * kernel_shared_local[(7)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(26)] * kernel_shared_local[(8)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(29)] * kernel_shared_local[(8)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(32)] * kernel_shared_local[(8)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(35)] * kernel_shared_local[(8)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(36)] * kernel_shared_local[(9)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(39)] * kernel_shared_local[(9)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(42)] * kernel_shared_local[(9)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(45)] * kernel_shared_local[(9)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(37)] * kernel_shared_local[(10)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(40)] * kernel_shared_local[(10)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(43)] * kernel_shared_local[(10)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(46)] * kernel_shared_local[(10)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(38)] * kernel_shared_local[(11)]));
    compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared_local[(41)] * kernel_shared_local[(11)]));
    compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared_local[(44)] * kernel_shared_local[(11)]));
    compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared_local[(47)] * kernel_shared_local[(11)]));
  }
  compute[(((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 224)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 224)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 56))] = compute_local[(1)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 224)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 112))] = compute_local[(2)];
  compute[((((((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 1568)) + (((int)threadIdx.y) * 224)) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)) + 168))] = compute_local[(3)];
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

    dim3 grid(7,2,4);
    dim3 block(8,7,8);

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

