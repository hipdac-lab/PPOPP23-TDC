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
  float compute_local[1];
  __shared__ float pad_temp_shared[288];
  __shared__ float kernel_shared[576];
  float pad_temp_shared_local[6];
  float kernel_shared_local[6];
  compute_local[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    if (((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) / 3)) >> 2) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 32) {
        if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 288) {
          if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 36) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)))] = ((((1 <= (((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3))) && ((((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) / 3)) < 15)) && (1 <= ((((int)blockIdx.x) * 7) + ((((int)threadIdx.x) % 3) * 3)))) ? data[(((((((((rc_outer * 1568) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 28)) + ((((int)threadIdx.x) / 3) * 14)) + (((int)blockIdx.x) * 7)) + ((((int)threadIdx.x) % 3) * 3)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 3) + 1) / 9)) >> 2) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 1) / 9)) < 32) {
        if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 287) {
          if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 35) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 1))] = (((((1 <= (((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 1) / 9))) && ((((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 1) / 9)) < 15)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 3) + 1) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 3) + 1) % 9)) < 15)) ? data[(((((((((rc_outer * 1568) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 28)) + ((((((int)threadIdx.x) * 3) + 1) / 9) * 14)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 3) + 1) % 9)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 3) + 2) / 9)) >> 2) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 2) / 9)) < 32) {
        if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) < 286) {
          if (((((int)threadIdx.y) * 18) + (((int)threadIdx.x) * 3)) < 34) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[(((((((int)threadIdx.z) * 36) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.x) * 3)) + 2))] = (((((1 <= (((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 2) / 9))) && ((((((int)blockIdx.y) * 2) + (((int)threadIdx.y) * 2)) + (((((int)threadIdx.x) * 3) + 2) / 9)) < 15)) && (1 <= ((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 3) + 2) % 9)))) && (((((int)blockIdx.x) * 7) + (((((int)threadIdx.x) * 3) + 2) % 9)) < 15)) ? data[(((((((((rc_outer * 1568) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 28)) + ((((((int)threadIdx.x) * 3) + 2) / 9) * 14)) + (((int)blockIdx.x) * 7)) + (((((int)threadIdx.x) * 3) + 2) % 9)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 576) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 72) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + ((((int)threadIdx.x) * 2) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 575) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 71) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 1))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + ((((int)threadIdx.x) * 2) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + ((((int)threadIdx.x) * 2) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 192) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 574) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 70) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 2))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + ((((int)threadIdx.x) * 2) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 573) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 69) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 3))] = kernel[((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((((int)threadIdx.x) * 2) + 1) * 3)))];
              }
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 572) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 68) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 4))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((((int)threadIdx.x) * 2) + 1) * 3)) + 1))];
              }
            }
          }
        }
      }
    }
    if (((((((int)threadIdx.y) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) >> 3) + ((int)threadIdx.z)) < 8) {
      if ((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
        if ((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) < 191) {
          if ((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) < 571) {
            if (((((int)threadIdx.y) * 36) + (((int)threadIdx.x) * 6)) < 67) {
              if (((int)threadIdx.x) < 6) {
                kernel_shared[(((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + 5))] = kernel[(((((((((int)blockIdx.z) * 2304) + (((int)threadIdx.z) * 288)) + (rc_outer * 72)) + (((int)threadIdx.y) * 36)) + (((((int)threadIdx.x) * 2) + 1) * 3)) + 2))];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    pad_temp_shared_local[(0)] = pad_temp_shared[(((((int)threadIdx.y) * 9) + ((int)threadIdx.x)))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 9))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 18))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 36))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 45))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 54))];
    kernel_shared_local[(0)] = kernel_shared[((((int)threadIdx.z) * 72))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 3))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 6))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 9))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 12))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 15))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 1))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 10))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 19))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 37))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 46))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 55))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 1))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 4))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 7))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 10))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 13))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 16))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 2))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 11))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 20))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 38))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 47))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 56))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 2))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 5))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 8))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 11))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 14))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 17))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 72))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 81))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 90))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 108))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 117))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 126))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 18))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 21))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 24))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 27))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 30))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 33))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 73))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 82))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 91))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 109))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 118))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 127))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 19))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 22))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 25))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 28))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 31))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 34))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 74))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 83))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 92))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 110))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 119))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 128))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 20))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 23))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 26))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 29))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 32))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 35))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 144))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 153))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 162))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 180))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 189))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 198))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 36))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 39))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 42))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 45))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 48))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 51))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 145))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 154))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 163))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 181))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 190))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 199))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 37))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 40))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 43))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 46))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 49))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 52))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 146))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 155))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 164))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 182))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 191))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 200))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 38))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 41))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 44))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 47))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 50))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 53))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 216))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 225))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 234))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 252))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 261))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 270))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 54))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 57))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 60))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 63))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 66))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 69))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 217))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 226))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 235))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 253))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 262))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 271))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 55))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 58))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 61))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 64))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 67))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 70))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
    pad_temp_shared_local[(0)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 218))];
    pad_temp_shared_local[(1)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 227))];
    pad_temp_shared_local[(2)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 236))];
    pad_temp_shared_local[(3)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 254))];
    pad_temp_shared_local[(4)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 263))];
    pad_temp_shared_local[(5)] = pad_temp_shared[((((((int)threadIdx.y) * 9) + ((int)threadIdx.x)) + 272))];
    kernel_shared_local[(0)] = kernel_shared[(((((int)threadIdx.z) * 72) + 56))];
    kernel_shared_local[(1)] = kernel_shared[(((((int)threadIdx.z) * 72) + 59))];
    kernel_shared_local[(2)] = kernel_shared[(((((int)threadIdx.z) * 72) + 62))];
    kernel_shared_local[(3)] = kernel_shared[(((((int)threadIdx.z) * 72) + 65))];
    kernel_shared_local[(4)] = kernel_shared[(((((int)threadIdx.z) * 72) + 68))];
    kernel_shared_local[(5)] = kernel_shared[(((((int)threadIdx.z) * 72) + 71))];
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(0)] * kernel_shared_local[(0)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(1)] * kernel_shared_local[(1)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(2)] * kernel_shared_local[(2)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(3)] * kernel_shared_local[(3)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(4)] * kernel_shared_local[(4)]));
    compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared_local[(5)] * kernel_shared_local[(5)]));
  }
  compute[(((((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)blockIdx.x) * 7)) + ((int)threadIdx.x)))] = compute_local[(0)];
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
    dim3 block(7,2,8);

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


