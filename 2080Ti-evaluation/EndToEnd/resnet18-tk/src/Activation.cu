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
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&activationOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(activationOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
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
/*int main(void){
    Activation activation;
    activation.initialize(1,112,112,64);
    float *input;
    float *hostInput = (float *)malloc((1*64*112*112)*sizeof(float));
    for(int i=0;i<1*64*112*112;++i){
        hostInput[i] = 0.000001f;
    }
    cudaMalloc(&input,1*64*112*112*sizeof(float));
    cudaMemcpy(input,hostInput,1*64*112*112*sizeof(float),cudaMemcpyHostToDevice);

    //conv.forward(input);
    //float *outputPython = load_input("../conv.bin",1*112*112*64);
    float *outputCudnn = (float *)malloc(1*112*112*64*sizeof(float));
    cudaMemcpy(outputCudnn,activation.forward(input),1*112*112*64*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<outputCudnn[63]<<endl;
    float diff = 0.0f;
    cout<<outputCudnn[63]<<endl;
    cout<<diff<<endl;
    return 0;
}*/
