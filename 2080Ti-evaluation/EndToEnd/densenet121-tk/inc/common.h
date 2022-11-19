//
// Created by lizhi on 3/19/22.
//
#ifndef TURKER_CONV_COMMON_H
#define TURKER_CONV_COMMON_H
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
using namespace std;
inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<endl;
        exit(-1);
    }
}
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
void load_input(string input_path,unsigned int dataSize,float *input);
class Conv{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int N;
    unsigned int PAD;
    unsigned int hOut;
    unsigned int wOut;
    unsigned int R;
    unsigned int S;
    bool use_bias;
    float *cpuKernel;
    float alpha = 1.0f;
    float beta = 0.0f;
    float beta2 = 1.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t biasDescriptor;
    float *output;
    float *kernel;
    float *bias;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
                    unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile, bool use_bias);
    float *forward(float *input);
};
class TkShape {
public:
    unsigned int C;
    unsigned int N1;
    unsigned int N2;
    unsigned int N3;
    bool down_sample;
    TkShape(unsigned c, unsigned n1, unsigned n2, unsigned n3);
};
class TkWeight {
public:
    string weight1;
    string weight2;
    string weight3;
    TkWeight(string w1, string w2, string w3);
};
class TkConv{
public:
    Conv conv1;
    Conv conv2;
    Conv conv3;
    void initialize(unsigned int C,unsigned int N1,unsigned int N2,unsigned int N3,unsigned int height, unsigned int width,
                    string conv1_weight,string conv2_weight,string conv3_weight, bool down_sample);
    float *forward(float *x);
};
class Pool{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    unsigned int hOut;
    unsigned int wOut;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t poolingCudnn;
    cudnnTensorDescriptor_t poolingInputDescriptor;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnTensorDescriptor_t poolingOutputDescriptor;
    float *output;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int pad,unsigned int windowH,unsigned int windowW,
                    cudnnPoolingMode_t mode,unsigned int stride);
    float * forward(float *input);
};
class BatchNorm{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t batchNormCudnn;
    cudnnTensorDescriptor_t batchNormInputDescriptor;
    cudnnTensorDescriptor_t batchNormOutputDescriptor;
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
    float *cpuKernel;
    float *scaleDev;
    float *shiftDev;
    float *meanDev;
    float *varDev;
    void initialize(unsigned int b,unsigned int c,unsigned int h, unsigned int w,string weightFile);
    float * forward(float *input);
    float *output;
};
class Activation{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t activationCudnn;
    cudnnActivationMode_t MODE;
    cudnnTensorDescriptor_t activationInputDescriptor;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t activationOutputDescriptor;
    float *output;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w);
    float *forward(float *input);
};
class Add{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C;
    unsigned int B;
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnHandle_t addCudnn;
    cudnnTensorDescriptor_t addInputDescriptor;
    cudnnTensorDescriptor_t addOutputDescriptor;
    void initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w);
    float *forward(float *x,float *y);
};
class Concate{
public:
    unsigned int H;
    unsigned int W;
    unsigned int C1;
    unsigned int B;
    unsigned int C2;
    float alpha = 1.0f;
    float beta = 1.0f;
    float *output;
    void initialize(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w);
    float *forward(float *x,float *y);
};
class TenDenseLayer{
public:
    BatchNorm bn1;
    Activation relu1;
    Conv conv1;
    BatchNorm bn2;
    Activation relu2;
    TkConv conv2;
    Concate concat;
    TenDenseLayer(unsigned int C,unsigned int N1,unsigned int N2, unsigned int H, unsigned int W, string bn1_weight,string conv1_weight,
                  string bn2_weight,string conv2_weight);
    float *forward(float *x);
};
class TenDenseTransition{
public:
    BatchNorm bn1;
    Activation relu1;
    Conv conv1;
    Pool pool1;
    TenDenseTransition(unsigned int C,unsigned int N1, unsigned int H, unsigned int W, string bn1_weight,string conv1_weight);
    float *forward(float *x);
};

#endif //TURKER_CONV_COMMON_H
