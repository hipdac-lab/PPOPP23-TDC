#include "../inc/common.h"
int main(int argc,char *argv[]){

    float *dInput;
    float *input = new float[3*224*224];
    Conv conv1;
    conv1.initialize(1,3,224,224,64,3,7,7,2," ", false);
    BatchNorm bn1;
    bn1.initialize(1,64,112,112," ");
    Activation relu1;
    relu1.initialize(1,64,112,112);
    Pool pool1;
    pool1.initialize(1,64,112,112,1,3,3,CUDNN_POOLING_MAX,2);

    TenDenseLayer block1_layer1(64,128,32,56,56," "," "," "," ");
    TenDenseLayer block1_layer2(96,128,32,56,56," "," "," "," ");
    TenDenseLayer block1_layer3(128,128,32,56,56," "," "," "," ");
    TenDenseLayer block1_layer4(160,128,32,56,56," "," "," "," ");
    TenDenseLayer block1_layer5(192,128,32,56,56," "," "," "," ");
    TenDenseLayer block1_layer6(224,128,32,56,56," "," "," "," ");
    TenDenseTransition transition1(256,128,56,56," "," ");

    TenDenseLayer block2_layer1(128,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer2(160,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer3(192,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer4(224,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer5(256,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer6(288,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer7(320,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer8(352,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer9(384,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer10(416,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer11(448,128,32,28,28," "," "," "," ");
    TenDenseLayer block2_layer12(480,128,32,28,28," "," "," "," ");
    TenDenseTransition transition2(512,256,28,28," "," ");

    TenDenseLayer block3_layer1(256,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer2(288,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer3(320,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer4(352,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer5(384,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer6(416,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer7(448,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer8(480,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer9(512,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer10(544,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer11(576,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer12(608,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer13(640,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer14(672,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer15(704,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer16(736,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer17(768,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer18(800,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer19(832,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer20(864,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer21(896,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer22(928,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer23(960,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer24(992,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer25(1024,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer26(1056,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer27(1088,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer28(1120,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer29(1152,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer30(1184,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer31(1216,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer32(1248,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer33(1280,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer34(1312,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer35(1344,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer36(1376,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer37(1408,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer38(1440,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer39(1472,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer40(1504,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer41(1536,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer42(1568,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer43(1660,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer44(1632,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer45(1664,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer46(1696,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer47(1728,128,32,14,14," "," "," "," ");
    TenDenseLayer block3_layer48(1760,128,32,14,14," "," "," "," ");
    TenDenseTransition transition3(1792,896,14,14," "," ");

    TenDenseLayer block4_layer1(896,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer2(928,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer3(960,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer4(992,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer5(1024,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer6(1056,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer7(1088,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer8(1120,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer9(1152,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer10(1184,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer11(1216,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer12(1248,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer13(1280,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer14(1312,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer15(1344,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer16(1376,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer17(1408,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer18(1440,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer19(1472,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer20(1504,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer21(1536,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer22(1568,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer23(1600,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer24(1632,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer25(1664,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer26(1696,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer27(1728,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer28(1760,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer29(1792,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer30(1824,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer31(1856,128,32,7,7," "," "," "," ");
    TenDenseLayer block4_layer32(1888,128,32,7,7," "," "," "," ");

    BatchNorm bn2;
    bn2.initialize(1,1920,7,7," ");
    Activation relu2;
    relu2.initialize(1,1920,7,7);
    Pool pool2;
    pool2.initialize(1,920,7,7,0,7,7,CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,1);

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaMalloc(&dInput,224*224*3*sizeof(float));
    float *y;
    y = conv1.forward(dInput);
    y = bn1.forward(y);
    y = relu1.forward(y);
    y = pool1.forward(y);
    y = block1_layer1.forward(y);
    y = block1_layer2.forward(y);
    y = block1_layer3.forward(y);
    y = block1_layer4.forward(y);
    y = block1_layer5.forward(y);
    y = block1_layer6.forward(y);
    y = transition1.forward(y);

    y = block2_layer1.forward(y);
    y = block2_layer2.forward(y);
    y = block2_layer3.forward(y);
    y = block2_layer4.forward(y);
    y = block2_layer5.forward(y);
    y = block2_layer6.forward(y);
    y = block2_layer7.forward(y);
    y = block2_layer8.forward(y);
    y = block2_layer9.forward(y);
    y = block2_layer10.forward(y);
    y = block2_layer11.forward(y);
    y = block2_layer12.forward(y);
    y = transition2.forward(y);

    y = block3_layer1.forward(y);
    y = block3_layer2.forward(y);
    y = block3_layer3.forward(y);
    y = block3_layer4.forward(y);
    y = block3_layer5.forward(y);
    y = block3_layer6.forward(y);
    y = block3_layer7.forward(y);
    y = block3_layer8.forward(y);
    y = block3_layer9.forward(y);
    y = block3_layer10.forward(y);
    y = block3_layer11.forward(y);
    y = block3_layer12.forward(y);
    y = block3_layer13.forward(y);
    y = block3_layer14.forward(y);
    y = block3_layer15.forward(y);
    y = block3_layer16.forward(y);
    y = block3_layer17.forward(y);
    y = block3_layer18.forward(y);
    y = block3_layer19.forward(y);
    y = block3_layer20.forward(y);
    y = block3_layer21.forward(y);
    y = block3_layer22.forward(y);
    y = block3_layer23.forward(y);
    y = block3_layer24.forward(y);
    y = block3_layer25.forward(y);
    y = block3_layer26.forward(y);
    y = block3_layer27.forward(y);
    y = block3_layer28.forward(y);
    y = block3_layer29.forward(y);
    y = block3_layer30.forward(y);
    y = block3_layer31.forward(y);
    y = block3_layer32.forward(y);
    y = block3_layer33.forward(y);
    y = block3_layer34.forward(y);
    y = block3_layer35.forward(y);
    y = block3_layer36.forward(y);
    y = block3_layer37.forward(y);
    y = block3_layer38.forward(y);
    y = block3_layer39.forward(y);
    y = block3_layer40.forward(y);
    y = block3_layer41.forward(y);
    y = block3_layer42.forward(y);
    y = block3_layer43.forward(y);
    y = block3_layer44.forward(y);
    y = block3_layer45.forward(y);
    y = block3_layer46.forward(y);
    y = block3_layer47.forward(y);
    y = block3_layer48.forward(y);
    y = transition3.forward(y);

    y = block4_layer1.forward(y);
    y = block4_layer2.forward(y);
    y = block4_layer3.forward(y);
    y = block4_layer4.forward(y);
    y = block4_layer5.forward(y);
    y = block4_layer6.forward(y);
    y = block4_layer7.forward(y);
    y = block4_layer8.forward(y);
    y = block4_layer9.forward(y);
    y = block4_layer10.forward(y);
    y = block4_layer11.forward(y);
    y = block4_layer12.forward(y);
    y = block4_layer13.forward(y);
    y = block4_layer14.forward(y);
    y = block4_layer15.forward(y);
    y = block4_layer16.forward(y);
    y = block4_layer17.forward(y);
    y = block4_layer18.forward(y);
    y = block4_layer19.forward(y);
    y = block4_layer20.forward(y);
    y = block4_layer21.forward(y);
    y = block4_layer22.forward(y);
    y = block4_layer23.forward(y);
    y = block4_layer24.forward(y);
    y = block4_layer25.forward(y);
    y = block4_layer26.forward(y);
    y = block4_layer27.forward(y);
    y = block4_layer28.forward(y);
    y = block4_layer29.forward(y);
    y = block4_layer30.forward(y);
    y = block4_layer31.forward(y);
    y = block4_layer32.forward(y);
    y = bn2.forward(y);
    y = relu2.forward(y);
    y = pool2.forward(y);
    chkerr(cudaDeviceSynchronize());
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
        y = conv1.forward(dInput);
        y = bn1.forward(y);
        y = relu1.forward(y);
        y = pool1.forward(y);
        y = block1_layer1.forward(y);
        y = block1_layer2.forward(y);
        y = block1_layer3.forward(y);
        y = block1_layer4.forward(y);
        y = block1_layer5.forward(y);
        y = block1_layer6.forward(y);
        y = transition1.forward(y);

        y = block2_layer1.forward(y);
        y = block2_layer2.forward(y);
        y = block2_layer3.forward(y);
        y = block2_layer4.forward(y);
        y = block2_layer5.forward(y);
        y = block2_layer6.forward(y);
        y = block2_layer7.forward(y);
        y = block2_layer8.forward(y);
        y = block2_layer9.forward(y);
        y = block2_layer10.forward(y);
        y = block2_layer11.forward(y);
        y = block2_layer12.forward(y);
        y = transition2.forward(y);

        y = block3_layer1.forward(y);
        y = block3_layer2.forward(y);
        y = block3_layer3.forward(y);
        y = block3_layer4.forward(y);
        y = block3_layer5.forward(y);
        y = block3_layer6.forward(y);
        y = block3_layer7.forward(y);
        y = block3_layer8.forward(y);
        y = block3_layer9.forward(y);
        y = block3_layer10.forward(y);
        y = block3_layer11.forward(y);
        y = block3_layer12.forward(y);
        y = block3_layer13.forward(y);
        y = block3_layer14.forward(y);
        y = block3_layer15.forward(y);
        y = block3_layer16.forward(y);
        y = block3_layer17.forward(y);
        y = block3_layer18.forward(y);
        y = block3_layer19.forward(y);
        y = block3_layer20.forward(y);
        y = block3_layer21.forward(y);
        y = block3_layer22.forward(y);
        y = block3_layer23.forward(y);
        y = block3_layer24.forward(y);
        y = block3_layer25.forward(y);
        y = block3_layer26.forward(y);
        y = block3_layer27.forward(y);
        y = block3_layer28.forward(y);
        y = block3_layer29.forward(y);
        y = block3_layer30.forward(y);
        y = block3_layer31.forward(y);
        y = block3_layer32.forward(y);
        y = block3_layer33.forward(y);
        y = block3_layer34.forward(y);
        y = block3_layer35.forward(y);
        y = block3_layer36.forward(y);
        y = block3_layer37.forward(y);
        y = block3_layer38.forward(y);
        y = block3_layer39.forward(y);
        y = block3_layer40.forward(y);
        y = block3_layer41.forward(y);
        y = block3_layer42.forward(y);
        y = block3_layer43.forward(y);
        y = block3_layer44.forward(y);
        y = block3_layer45.forward(y);
        y = block3_layer46.forward(y);
        y = block3_layer47.forward(y);
        y = block3_layer48.forward(y);
        y = transition3.forward(y);

        y = block4_layer1.forward(y);
        y = block4_layer2.forward(y);
        y = block4_layer3.forward(y);
        y = block4_layer4.forward(y);
        y = block4_layer5.forward(y);
        y = block4_layer6.forward(y);
        y = block4_layer7.forward(y);
        y = block4_layer8.forward(y);
        y = block4_layer9.forward(y);
        y = block4_layer10.forward(y);
        y = block4_layer11.forward(y);
        y = block4_layer12.forward(y);
        y = block4_layer13.forward(y);
        y = block4_layer14.forward(y);
        y = block4_layer15.forward(y);
        y = block4_layer16.forward(y);
        y = block4_layer17.forward(y);
        y = block4_layer18.forward(y);
        y = block4_layer19.forward(y);
        y = block4_layer20.forward(y);
        y = block4_layer21.forward(y);
        y = block4_layer22.forward(y);
        y = block4_layer23.forward(y);
        y = block4_layer24.forward(y);
        y = block4_layer25.forward(y);
        y = block4_layer26.forward(y);
        y = block4_layer27.forward(y);
        y = block4_layer28.forward(y);
        y = block4_layer29.forward(y);
        y = block4_layer30.forward(y);
        y = block4_layer31.forward(y);
        y = block4_layer32.forward(y);
        chkerr(cudaDeviceSynchronize());
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float temp_time;
        cudaEventElapsedTime(&temp_time, event_start, event_stop);
        inference_time += temp_time;
    }
    ofstream outfile;
    char buffer[1000];
    int ret = sprintf(buffer,"%s,%f\n","densenet201-tk",inference_time/100);
    outfile.open("../../../../evaluation_outcome/2080Ti-end2end.csv", std::ios_base::app);
    outfile << buffer;
    return 0;
}
