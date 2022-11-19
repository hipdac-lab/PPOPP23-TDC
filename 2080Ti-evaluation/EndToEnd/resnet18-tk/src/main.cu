#include "../inc/common.h"
int main(int argc,char *argv[]){
    float *input = (float *)malloc(224*224*3*sizeof(float));
    //string imagePath = argv[1];
    float *dInput;
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    Conv conv1_conv;
    conv1_conv.initialize(1,3,224,224,64,3,7,7,2," ", false);
    BatchNorm conv1_bn;
    conv1_bn.initialize(1,64,112,112," ");
    Activation conv1_relu;
    conv1_relu.initialize(1,64,112,112);
    Pool conv1_max_pool;
    conv1_max_pool.initialize(1,64,112,112,1,3,3,CUDNN_POOLING_MAX,2);

    /*
     * layer 1
     */
    TkShape layer1_basic0_tk0_shape(64,64,32,64);
    TkShape layer1_basic0_tk1_shape(64,32,32,64);
    TkWeight layer1_basic0_tk0_weight(" "," "," ");
    TkWeight layer1_basic0_tk1_weight(" "," "," ");

    TkShape layer1_basic1_tk0_shape(64,32,32,64);
    TkShape layer1_basic1_tk1_shape(64,32,32,64);
    TkWeight layer1_basic1_tk0_weight(" "," "," ");
    TkWeight layer1_basic1_tk1_weight(" "," "," ");

    BasicBlock layer1_basic0(layer1_basic0_tk0_shape,layer1_basic0_tk1_shape,
                             layer1_basic0_tk0_weight,layer1_basic0_tk1_weight,
                             56,56," "," ");
    BasicBlock layer1_basic1(layer1_basic1_tk0_shape,layer1_basic1_tk1_shape,
                             layer1_basic1_tk0_weight,layer1_basic1_tk1_weight,
                             56,56," "," ");
    /*
     * layer 2
     */
    TkShape layer2_basic0_tk0_shape(64,64,64,128);
    TkShape layer2_basic0_tk1_shape(128,96,64,128);
    TkWeight layer2_basic0_tk0_weight(" "," "," ");
    TkWeight layer2_basic0_tk1_weight(" "," "," ");

    TkShape layer2_basic1_tk0_shape(128,96,64,128);
    TkShape layer2_basic1_tk1_shape(128,96,64,128);
    TkWeight layer2_basic1_tk0_weight(" "," "," ");
    TkWeight layer2_basic1_tk1_weight(" "," "," ");

    BasicBlock_Downsample layer2_basic0(layer2_basic0_tk0_shape,layer2_basic0_tk1_shape,
                                        layer2_basic0_tk0_weight,layer2_basic0_tk1_weight,
                                        56,56,128,
                                        " "," "," "," ");
    BasicBlock layer2_basic1(layer2_basic1_tk0_shape,layer2_basic1_tk1_shape,
                             layer2_basic1_tk0_weight,
                             layer2_basic1_tk1_weight,28,28,
                             " "," ");
    /*
     * layer 3
     */
    TkShape layer3_basic0_tk0_shape(128,96,96,256);
    TkShape layer3_basic0_tk1_shape(256,128,96,256);
    TkWeight layer3_basic0_tk0_weight(" "," "," ");
    TkWeight layer3_basic0_tk1_weight(" "," "," ");

    TkShape layer3_basic1_tk0_shape(256,128,96,256);
    TkShape layer3_basic1_tk1_shape(256,128,96,256);
    TkWeight layer3_basic1_tk0_weight(" "," "," ");
    TkWeight layer3_basic1_tk1_weight(" "," "," ");

    BasicBlock_Downsample layer3_basic0(layer3_basic0_tk0_shape,layer3_basic0_tk1_shape,
                                        layer3_basic0_tk0_weight,layer3_basic0_tk1_weight,
                                        28,28,256,
                                        " "," "," "," ");
    BasicBlock layer3_basic1(layer3_basic1_tk0_shape,layer3_basic1_tk1_shape,
                             layer3_basic1_tk0_weight,
                             layer3_basic1_tk1_weight,14,14,
                             " "," ");

    /*
     * layer 4
     */
    TkShape layer4_basic0_tk0_shape(256,160,160,512);
    TkShape layer4_basic0_tk1_shape(512,192,160,512);
    TkWeight layer4_basic0_tk0_weight(" "," "," ");
    TkWeight layer4_basic0_tk1_weight(" "," "," ");

    TkShape layer4_basic1_tk0_shape(512,192,160,512);
    TkShape layer4_basic1_tk1_shape(512,192,160,512);
    TkWeight layer4_basic1_tk0_weight(" "," "," ");
    TkWeight layer4_basic1_tk1_weight(" "," "," ");

    BasicBlock_Downsample layer4_basic0(layer4_basic0_tk0_shape,layer4_basic0_tk1_shape,
                                        layer4_basic0_tk0_weight,layer4_basic0_tk1_weight,
                                        14,14,512,
                                        " "," "," "," ");
    BasicBlock layer4_basic1(layer4_basic1_tk0_shape,layer4_basic1_tk1_shape,
                             layer4_basic1_tk0_weight,
                             layer4_basic1_tk1_weight,7,7,
                             " "," ");

    Pool avg_pool;
    avg_pool.initialize(1,512,7,7,0,7,7,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,1);
    Conv predict;
    predict.initialize(1,512,1,1,1000,0,1,1,1,"../../../weights/resnet50/weights/predictions.bin", true);

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    //cudaMemcpy(dInput, input, 224 * 224 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    float *out;
    out = conv1_conv.forward(dInput);
    out = conv1_bn.forward(out);
    out = conv1_relu.forward(out);
    out = conv1_max_pool.forward(out);
    out = layer1_basic0.forward(out);
    out = layer1_basic1.forward(out);
    out = layer2_basic0.forward(out);
    out = layer2_basic1.forward(out);
    out = layer3_basic0.forward(out);
    out = layer3_basic1.forward(out);
    out = layer4_basic0.forward(out);
    out = layer4_basic1.forward(out);
    out = predict.forward(out);
    chkerr(cudaDeviceSynchronize());
    cudaDeviceSynchronize();
    //cout<<"network construction finished"<<endl;
    float inference_time = 0.0f;
    for(int i=0;i<100;++i){
        time_t t;
        srand((unsigned) time(&t));
        for(int j =0;j<3*224*224;++j){
            input[j] = rand() % 10;
        }
        cudaMemcpy(dInput,input,3*224*224*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(event_start);
        out = conv1_conv.forward(dInput);
        out = conv1_bn.forward(out);
        out = conv1_relu.forward(out);
        out = conv1_max_pool.forward(out);
        out = layer1_basic0.forward(out);
        out = layer1_basic1.forward(out);
        out = layer2_basic0.forward(out);
        out = layer2_basic1.forward(out);
        out = layer3_basic0.forward(out);
        out = layer3_basic1.forward(out);
        out = layer4_basic0.forward(out);
        out = layer4_basic1.forward(out);
        out = predict.forward(out);
        chkerr(cudaDeviceSynchronize());
        cudaDeviceSynchronize();
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    ofstream outfile;
    char buffer[1000];
    int ret = sprintf(buffer,"%s,%f\n","resnet18-tk",inference_time/100);
    outfile.open("../../../../evaluation_outcome/2080Ti-end2end.csv", std::ios_base::app);
    outfile << buffer;
    return 0;
}
