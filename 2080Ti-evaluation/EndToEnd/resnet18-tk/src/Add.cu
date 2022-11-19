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
/*int main(void){
    Add add;
    add.initialize(1,112,112,64);
    float *input;
    float *input2;
    float *hostInput = (float *)malloc((1*64*112*112)*sizeof(float));
    for(int i=0;i<1*64*112*112;++i){
        hostInput[i] = 1.0f;
    }
    cudaMalloc(&input,1*64*112*112*sizeof(float));
    cudaMemcpy(input,hostInput,1*64*112*112*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc(&input2,1*64*112*112*sizeof(float));
    cudaMemcpy(input2,hostInput,1*64*112*112*sizeof(float),cudaMemcpyHostToDevice);
    //conv.forward(input);
    //float *outputPython = load_input("../conv.bin",1*112*112*64);
    float *outputCudnn = (float *)malloc(1*112*112*64*sizeof(float));
    cudaMemcpy(outputCudnn,add.forward(input,input2),1*112*112*64*sizeof(float),cudaMemcpyDeviceToHost);
    cout<<outputCudnn[63]<<endl;
    float diff = 0.0f;
    for(int i=0;i<112*112*64;i++){
        diff +=(outputCudnn[i] - outputPython[i]);
    }
    cout<<outputCudnn[63]<<" "<<outputPython[0]<<endl;
    cout<<diff<<endl;
    return 0;
}*/
