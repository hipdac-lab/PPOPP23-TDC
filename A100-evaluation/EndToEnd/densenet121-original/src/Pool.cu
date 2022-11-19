#include "../inc/common.h"
/*
 *  available pooling mode
 *  CUDNN_POOLING_MAX,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
 */
void Pool::initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int pad,unsigned int windowH,unsigned int windowW,
           cudnnPoolingMode_t mode,unsigned int stride) {
    B = b;
    C = c;
    H = h;
    W = w;
    hOut = (h - windowH + 2*pad)/stride + 1;
    wOut = (w - windowW + 2*pad)/stride + 1;
    checkCUDNN(cudnnCreate(&poolingCudnn));
    checkCUDNN(cudnnCreateTensorDescriptor(&poolingInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(poolingInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&poolingOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(poolingOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/hOut,
            /*image_width=*/wOut));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
    cudnnSetPooling2dDescriptor(poolingDesc,mode,CUDNN_NOT_PROPAGATE_NAN,windowH,windowW,
                                pad,pad,stride,stride);
    cudaMalloc(&output,B*C*hOut*wOut*sizeof(float));
}
float * Pool::forward(float *input) {
    checkCUDNN(cudnnPoolingForward(
            poolingCudnn,
            poolingDesc,
            &alpha,
            poolingInputDescriptor,
            input,
            &beta,
            poolingOutputDescriptor,
            output));
    return output;
}
