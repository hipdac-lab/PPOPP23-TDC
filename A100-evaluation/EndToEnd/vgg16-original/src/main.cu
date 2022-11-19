#include "../inc/common.h"
int main(int argc,char *argv[]){
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *input = new float[3*224*224];
    Conv_blk conv_blk1(3,64,64,224,224," "," "," "," ");
    Pool pool1;
    pool1.initialize(1,64,224,224,0,2,2,CUDNN_POOLING_MAX,2);

    Conv_blk conv_blk2(64,128,128,112,112," "," "," "," ");
    Pool pool2;
    pool2.initialize(1,128,112,112,0,2,2,CUDNN_POOLING_MAX,2);

    Conv_blk3 conv_blk3(128,256,256,256,56,56," ",
                        " "," "," "," "," ");
    Pool pool3;
    pool3.initialize(1,256,56,56,0,2,2,CUDNN_POOLING_MAX,2);

    Conv_blk3 conv_blk4(256,512,512,512,28,28," ",
                        " "," "," "," "," ");
    Pool pool4;
    pool4.initialize(1,512,28,28,0,2,2,CUDNN_POOLING_MAX,2);

    Conv_blk3 conv_blk5(512,512,512,512,14,14," ",
                        " "," "," "," "," ");
    Pool pool5;
    pool5.initialize(1,512,14,14,0,2,2,CUDNN_POOLING_MAX,2);

    Conv conv1;
    conv1.initialize(1,512,7,7,4096,0,7,7,1," ", false);

    Activation relu1;
    relu1.initialize(1,4096,1,1);

    Conv conv2;
    conv2.initialize(1,4096,1,1,4096,0,1,1,1," ", false);

    Activation relu2;
    relu2.initialize(1,4096,1,1);

    Conv conv3;
    conv3.initialize(1,4096,1,1,1000,0,1,1,1," ", false);

    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    float *y;
    y = conv_blk1.forward(dInput);
    y = pool1.forward(y);
    y = conv_blk2.forward(y);
    y = pool2.forward(y);
    y = conv_blk3.forward(y);
    y = pool3.forward(y);
    y = conv_blk4.forward(y);
    y = pool4.forward(y);
    y = conv_blk5.forward(y);
    y = pool5.forward(y);
    y = conv1.forward(y);
    y = relu1.forward(y);
    y = conv2.forward(y);
    y = relu2.forward(y);
    y = conv3.forward(y);
    chkerr(cudaDeviceSynchronize());
    cudaDeviceSynchronize();
    float inference_time = 0.0f;
    for(int i=0;i<100;++i){
        time_t t;
        srand((unsigned) time(&t));
        for(int j =0;j<3*224*224;++j){
            input[j] = rand() % 10;
        }
        cudaMemcpy(dInput,input,3*224*224*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        y = conv_blk1.forward(dInput);
        y = pool1.forward(y);
        y = conv_blk2.forward(y);
        y = pool2.forward(y);
        y = conv_blk3.forward(y);
        y = pool3.forward(y);
        y = conv_blk4.forward(y);
        y = pool4.forward(y);
        y = conv_blk5.forward(y);
        y = pool5.forward(y);
        y = conv1.forward(y);
        y = relu1.forward(y);
        y = conv2.forward(y);
        y = relu2.forward(y);
        y = conv3.forward(y);
        chkerr(cudaDeviceSynchronize());
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    ofstream outfile;
    char buffer[1000];
    int ret = sprintf(buffer,"%s,%f\n","vgg16",inference_time/100);
    outfile.open("../../../../evaluation_outcome/A100-end2end.csv", std::ios_base::app);
    outfile << buffer;
    return 0;
}
