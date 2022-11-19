#include "../inc/common.h"
int main(int argc,char *argv[]){
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    float *input = new float[3*224*224];
    Conv conv1;
    conv1.initialize(1,3,224,224,64,1,3,3,1," ", false);
    BatchNorm bn1;
    bn1.initialize(1,64,224,224," ");
    Activation relu1;
    relu1.initialize(1,64,224,224);
    TkConv conv2;
    conv2.initialize(64,64,32,64,224,224," "," "," ", false);
    BatchNorm bn2;
    bn2.initialize(1,64,224,224," ");
    Activation relu2;
    relu2.initialize(1,64,224,224);
    Pool pool1;
    pool1.initialize(1,64,224,224,0,2,2,CUDNN_POOLING_MAX,2);

    BLkShape shape1(64,64,32,128,64,32,128);
    Conv_blk conv_blk2(shape1,112,112," "," "," "," ");
    Pool pool2;
    pool2.initialize(1,128,112,112,0,2,2,CUDNN_POOLING_MAX,2);

    BLkShape3 shape2(128,64,32,256,64,32,256,64,64,256);
    Conv_blk3 conv_blk3(shape2,56,56," ",
                        " "," "," "," "," ");
    Pool pool3;
    pool3.initialize(1,256,56,56,0,2,2,CUDNN_POOLING_MAX,2);

    BLkShape3 shape3(256,160,96,512,160,96,512,192,96,512);
    Conv_blk3 conv_blk4(shape3,28,28," ",
                        " "," "," "," "," ");
    Pool pool4;
    pool4.initialize(1,512,28,28,0,2,2,CUDNN_POOLING_MAX,2);

    BLkShape3 shape4(512,192,96,512,192,96,512,192,96,512);
    Conv_blk3 conv_blk5(shape4,14,14," ",
                        " "," "," "," "," ");
    Pool pool5;
    pool5.initialize(1,512,14,14,0,2,2,CUDNN_POOLING_MAX,2);

    Conv fc1_conv1;
    fc1_conv1.initialize(1,512,7,7,288,0,1,1,1," ", false);
    Conv fc1_conv2;
    fc1_conv2.initialize(1,288,7,7,288,0,7,7,1," ", false);
    Conv fc1_conv3;
    fc1_conv3.initialize(1,288,1,1,4096,0,1,1,1," ", false);

    Activation fc1_relu;
    fc1_relu.initialize(1,4096,1,1);

    Conv fc2_conv1;
    fc2_conv1.initialize(1,4096,1,1,512,0,1,1,1," ", false);
    Conv fc2_conv2;
    fc2_conv2.initialize(1,512,1,1,4096,0,1,1,1," ", false);
    Activation fc2_relu;
    fc2_relu.initialize(1,4096,1,1);

    Conv fc;
    fc.initialize(1,4096,1,1,1000,0,1,1,1," ", true);
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    float *y;
    y = conv1.forward(dInput);
    y = bn1.forward(y);
    y = relu1.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    y = relu2.forward(y);
    y = pool1.forward(y);
    y = conv_blk2.forward(y);
    y = pool2.forward(y);
    y = conv_blk3.forward(y);
    y = pool3.forward(y);
    y = conv_blk4.forward(y);
    y = pool4.forward(y);
    y = conv_blk5.forward(y);
    y = pool5.forward(y);
    y = fc1_conv1.forward(y);
    y = fc1_conv2.forward(y);
    y = fc1_conv3.forward(y);
    y = fc1_relu.forward(y);
    y = fc2_conv1.forward(y);
    y = fc2_conv2.forward(y);
    y = fc2_relu.forward(y);
    y = fc.forward(y);
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
        y = conv1.forward(dInput);
        y = bn1.forward(y);
        y = relu1.forward(y);
        y = conv2.forward(y);
        y = bn2.forward(y);
        y = relu2.forward(y);
        y = pool1.forward(y);
        y = conv_blk2.forward(y);
        y = pool2.forward(y);
        y = conv_blk3.forward(y);
        y = pool3.forward(y);
        y = conv_blk4.forward(y);
        y = pool4.forward(y);
        y = conv_blk5.forward(y);
        y = pool5.forward(y);
        y = fc1_conv1.forward(y);
        y = fc1_conv2.forward(y);
        y = fc1_conv3.forward(y);
        y = fc1_relu.forward(y);
        y = fc2_conv1.forward(y);
        y = fc2_conv2.forward(y);
        y = fc2_relu.forward(y);
        y = fc.forward(y);
        chkerr(cudaDeviceSynchronize());
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    ofstream outfile;
    char buffer[1000];
    int ret = sprintf(buffer,"%s,%f\n","vgg16-tk",inference_time/100);
    outfile.open("../../../../evaluation_outcome/A100-end2end.csv", std::ios_base::app);
    outfile << buffer;
    return 0;
}
