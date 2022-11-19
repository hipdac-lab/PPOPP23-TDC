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
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&poolingOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(poolingOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
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
/*int main(void){
    Pool pool;
    pool.initialize(1,112,112,64,1,3,3,CUDNN_POOLING_MAX,2);
    float *input;
    float *hostInput = (float *)malloc((1*64*112*112)*sizeof(float));
    for(int i=0;i<1*64*112*112;++i){
        hostInput[i] = 1.0f;
    }
    cudaMalloc(&input,1*64*112*112*sizeof(float));
    cudaMemcpy(input,hostInput,1*64*112*112*sizeof(float),cudaMemcpyHostToDevice);

    //conv.forward(input);
    //float *outputPython = load_input("../conv.bin",1*112*112*64);
    float *outputCudnn = (float *)malloc(1*112*112*64*sizeof(float));
    cudaMemcpy(outputCudnn,pool.forward(input),1*56*56*64*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<outputCudnn[63]<<endl;
    float diff = 0.0f;
    for(int i=0;i<112*112*64;i++){
        diff +=(outputCudnn[i] - outputPython[i]);
    }
    cout<<outputCudnn[63]<<" "<<outputPython[0]<<endl;
    cout<<diff<<endl;
    return 0;
}*/