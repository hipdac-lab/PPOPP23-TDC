#include "../inc/common.h"
void BatchNorm::initialize(unsigned int b, unsigned int c, unsigned int h, unsigned int w,string weight) {
    B = b;
    H = h;
    W = w;
    C = c;
    chkerr(cudaMalloc(&scaleDev,C*sizeof(float)));
    chkerr(cudaMalloc(&shiftDev,C*sizeof(float)));
    chkerr(cudaMalloc(&meanDev,C*sizeof(float)));
    chkerr(cudaMalloc(&varDev,C*sizeof(float)));
    checkCUDNN(cudnnCreate(&batchNormCudnn));
    cudaMalloc(&output,B*C*H*W*sizeof(float));
    checkCUDNN(cudnnCreateTensorDescriptor(&batchNormInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(batchNormInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&batchNormOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(batchNormOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
            /*channels=*/C,
            /*image_height=*/H,
            /*image_width=*/W));
    checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/C,
            /*image_height=*/1,
            /*image_width=*/1));

    this->cpuKernel = (float *)malloc(4*C*sizeof(float));
    //load_input(weight,4*C,cpuKernel);
    /*try{
        load_input(weight,4*C,cpuKernel);
    }catch (const char* msg) {
        cerr << msg << endl;
    }*/
    for(int i=0;i<4*C;++i){
        this->cpuKernel[i] = 1.0f;
    }
    chkerr(cudaMemcpy(scaleDev,cpuKernel,C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(shiftDev,&cpuKernel[C],C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(meanDev,&cpuKernel[2*C],C*sizeof(float),cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(varDev,&cpuKernel[3*C],C*sizeof(float),cudaMemcpyHostToDevice));
    free(cpuKernel);
}
float * BatchNorm::forward(float *input) {
    checkCUDNN(cudnnBatchNormalizationForwardInference(
            batchNormCudnn,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            batchNormInputDescriptor,
            input, //gpu上的
            batchNormOutputDescriptor,
            output, //gpu上的
            bnScaleBiasMeanVarDesc,
            scaleDev,  //gpu上的
            shiftDev,    //gpu上的
            meanDev,  //gpu上的
            varDev,//gpu上的
            CUDNN_BN_MIN_EPSILON
    ));
    return output;
}
