#include "../inc/common.h"
/*
 * cudnnNanPropagation_t : CUDNN_NOT_PROPAGATE_NAN,CUDNN_PROPAGATE_NAN
 * cudnnActivationMode_t:
 * CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_CLIPPED_RELU,
    CUDNN_ACTIVATION_ELU,
 *
 */
void Activation::initialize(unsigned int b, unsigned int c, unsigned int h, unsigned int w) {
    B = b;
    C = c;
    H = h;
    W = w;
    cudaMalloc(&output,B*C*H*W*sizeof(float));
    checkCUDNN(cudnnCreate(&activationCudnn));
    checkCUDNN(cudnnCreateTensorDescriptor(&activationInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(activationInputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&activationOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(activationOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activationDesc,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0.0f));
}
float * Activation::forward(float *input) {
    checkCUDNN(cudnnActivationForward(activationCudnn,activationDesc,&alpha,activationInputDescriptor,input,&beta,activationOutputDescriptor,output));
    return output;
}
