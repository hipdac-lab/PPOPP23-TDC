#include "../inc/common.h"
void Conv::initialize(unsigned int b,unsigned int c,unsigned int h,unsigned int w,unsigned int n,
           unsigned int pad,unsigned int r,unsigned int s,unsigned int stride,string weightFile, bool use_bias){
    this->B = b;
    this->C = c;
    this->H = h;
    this->W = w;
    this->N = n;
    this->R = r;
    this->S = s;
    this->hOut = (H+2*pad - r)/stride + 1;
    this->wOut = (W+2*pad - s)/stride + 1;
    cudaMalloc(&kernel,sizeof(float)*C*N*R*S);
    cudaMalloc(&bias,sizeof(float)*N);
    cudaMalloc(&this->output,sizeof(float)*B*hOut*wOut*N);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnCreateTensorDescriptor(&biasDescriptor);
    cudnnSetTensor4dDescriptor(biasDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/1,
            /*channels=*/N,
            /*image_height=*/1,
            /*image_width=*/1);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/B,
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
            /*pad_height=*/pad,
            /*pad_width=*/pad,
            /*vertical_stride=*/stride,
            /*horizontal_stride=*/stride,
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
            /*batch_size=*/B,
            /*channels=*/N,
            /*image_height=*/hOut,
            /*image_width=*/wOut);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                            &workspace_bytes);
    cudaMalloc(&d_workspace, workspace_bytes);
    if(use_bias){
        this->use_bias = true;
        unsigned int kernelSize = R*S*C*N + N;//kernel + bias
        this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
        /*try{
            load_input(weightFile,kernelSize,cpuKernel);
        }catch (const char* msg) {
            cerr << msg << endl;
        }*/
        for(int i=0;i<kernelSize;++i){
            this->cpuKernel[i] = 1.0f;
        }
        cudaMemcpy(kernel,cpuKernel,R*S*C*N*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(bias,&cpuKernel[R*S*C*N],N*sizeof(float),cudaMemcpyHostToDevice);
        free(cpuKernel);
    }else{
        this->use_bias = false;
        unsigned int kernelSize = R*S*C*N;//kernel + bias
        this->cpuKernel = (float *)malloc(kernelSize*sizeof(float));
        /*try{
            load_input(weightFile,kernelSize,cpuKernel);
        }catch (const char* msg) {
            cerr << msg << endl;
        }*/
        for(int i=0;i<kernelSize;++i){
            this->cpuKernel[i] = 1.0f;
        }
        cudaMemcpy(kernel,cpuKernel,R*S*C*N*sizeof(float),cudaMemcpyHostToDevice);
        free(cpuKernel);
    }
}
float * Conv::forward(float *input) {
    cudaMemset(output, 0, B*N*hOut*wOut*sizeof(float));
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
    if(use_bias){
        checkCUDNN(cudnnAddTensor(convCudnn,&alpha,biasDescriptor,bias,&beta2,convOutputDescriptor,output));
    }
    return output;
}
__global__ void concate(unsigned int b,unsigned int c1,unsigned int c2,unsigned int h,unsigned int w,float *x,
                        float * y, float *z){
    unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id >=b*(c1+c2)*h*w){
        return ;
    }
    for(unsigned int i = id;i<b*h*w*(c1+c2);i+=gridDim.x*blockDim.x){
        unsigned int hw = i %(h*w);
        unsigned int c = i / (h*w);
        if(c >= c1){
            float v = y[(c - c1)*h*w+hw];
            z[c*h*w+hw] = v;
        }else{
            float v = x[c * h * w+hw];
            z[c*h*w+hw] = v;
        }
    }
}
void Concate::initialize(unsigned int b, unsigned int c1, unsigned int c2, unsigned int h, unsigned int w) {
    B = b;
    C1 = c1;
    C2 = c2;
    H = h;
    W = w;
    cudaMalloc(&output,b*(c1+c2)*h*w*sizeof(float));
}
float * Concate::forward(float *x, float *y) {
    concate<<<84,1024>>>(B,C1,C2,H,W,x,y,output);
    return output;
}
