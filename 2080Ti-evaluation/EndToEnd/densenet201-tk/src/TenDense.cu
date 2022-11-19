

#include "../inc/common.h"
TkWeight::TkWeight(string w1, string w2, string w3) {
    this->weight1 = w1;
    this->weight2 = w2;
    this->weight3 = w3;
}
TkShape::TkShape(unsigned int c, unsigned int n1, unsigned int n2, unsigned int n3) {
    this->C = c;
    this->N1 = n1;
    this->N2 = n2;
    this->N3 = n3;
}
void TkConv::initialize(unsigned int C, unsigned int N1, unsigned int N2, unsigned int N3, unsigned int height,
                        unsigned int width, string conv1_weight, string conv2_weight, string conv3_weight, bool down_sample) {
    conv1.initialize(1,C,height,width,N1,0,1,1,1,conv1_weight,false);
    if(down_sample){
        conv2.initialize(1,N1,height,width,N2,1,3,3,2,conv2_weight, true);
        conv3.initialize(1,N2,height/2,width/2,N3,0,1,1,1,conv3_weight, false);
    }else{
        conv2.initialize(1,N1,height,width,N2,1,3,3,1,conv2_weight, true);
        conv3.initialize(1,N2,height,width,N3,0,1,1,1,conv3_weight, false);
    }
}

float *TkConv::forward(float *x) {
    float *y = conv1.forward(x);
    y = conv2.forward(y);
    y = conv3.forward(y);
    return y;
}
TenDenseLayer::TenDenseLayer(unsigned int C,unsigned int N1, unsigned int N2,unsigned int H, unsigned int W,
                             string bn1_weight, string conv1_weight, string bn2_weight, string conv2_weight) {

    bn1.initialize(1,C,H,W," ");
    relu1.initialize(1,C,H,W);
    conv1.initialize(1,C,H,W,128,0,1,1,1," ", false);
    bn2.initialize(1,128,H,W," ");
    relu2.initialize(1,128,H,W);
    conv2.initialize(128,64,32,32,H,W," "," "," ", false);
    concat.initialize(1,C,32,H,W);
}
float *TenDenseLayer::forward(float *x) {
    float *y = bn1.forward(x);
    y = relu1.forward(y);
    y = conv1.forward(y);
    y = bn2.forward(y);
    y = relu2.forward(y);
    y = conv2.forward(y);
    y = concat.forward(x,y);
    return y;
}
TenDenseTransition::TenDenseTransition(unsigned int C, unsigned int N1, unsigned int H, unsigned int W,
                                       string bn1_weight, string conv1_weight) {
    bn1.initialize(1,C,H,W," ");
    relu1.initialize(1,C,H,W);
    conv1.initialize(1,C,H,W,N1,0,1,1,1," ", false);
    pool1.initialize(1,N1,H,W,0,2,2,CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,2);
}
float * TenDenseTransition::forward(float *x) {
    float *y = bn1.forward(x);
    y = relu1.forward(y);
    y = conv1.forward(y);
    y = pool1.forward(y);
    return y;
}