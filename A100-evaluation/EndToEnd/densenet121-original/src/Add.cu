#include "../inc/common.h"
void Add::initialize(unsigned int b, unsigned int c, unsigned int h, unsigned int w) {
    B = b;
    C = c;
    H = h;
    W = w;
    checkCUDNN(cudnnCreate(&addCudnn));
    checkCUDNN(cudnnCreateTensorDescriptor(&addInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(addInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&addOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(addOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
}
float *Add::forward(float *x, float *y) {
    checkCUDNN(cudnnAddTensor(addCudnn,&alpha,addInputDescriptor,x,&beta,addOutputDescriptor,y));
    return y;
}
