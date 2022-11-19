
#include "../inc/common.h"

BLkShape3::BLkShape3(unsigned int c, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4,
                     unsigned int n5, unsigned int n6, unsigned int n7, unsigned int n8, unsigned int n9) {
    this->C = c;
    this->N1 = n1;
    this->N2 = n2;
    this->N3 = n3;
    this->N4 = n4;
    this->N5 = n5;
    this->N6 = n6;
    this->N7 = n7;
    this->N8 = n8;
    this->N9 = n9;
}
BLkShape::BLkShape(unsigned int c, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4,
                     unsigned int n5, unsigned int n6) {
    this->C = c;
    this->N1 = n1;
    this->N2 = n2;
    this->N3 = n3;
    this->N4 = n4;
    this->N5 = n5;
    this->N6 = n6;
}
void TkConv::initialize(unsigned int C, unsigned int N1, unsigned int N2, unsigned int N3, unsigned int height,
                        unsigned int width, string conv1_weight, string conv2_weight, string conv3_weight, bool down_sample) {
    conv1.initialize(1,C,height,width,N1,0,1,1,1,conv1_weight,false);
    if(down_sample){
        conv2.initialize(1,N1,height,width,N2,1,3,3,2,conv2_weight, false);
        conv3.initialize(1,N2,height/2,width/2,N3,0,1,1,1,conv3_weight, false);
    }else{
        conv2.initialize(1,N1,height,width,N2,1,3,3,1,conv2_weight, false);
        conv3.initialize(1,N2,height,width,N3,0,1,1,1,conv3_weight, false);
    }
}

float *TkConv::forward(float *x) {
    float *y = conv1.forward(x);
    y = conv2.forward(y);
    y = conv3.forward(y);
    return y;
}
Conv_blk::Conv_blk(BLkShape shape,unsigned int height, unsigned int width,
                   string conv_weight1,string bn_weight1,string conv_weight2, string bn_weight2) {
    conv1.initialize(shape.C,shape.N1,shape.N2,shape.N3,height,width," "," "," ", false);
    bn1.initialize(1,shape.N3,height,width,bn_weight1);
    relu1.initialize(1,shape.N3,height,width);

    conv2.initialize(shape.N3,shape.N4,shape.N5,shape.N6,height,width," "," "," ", false);
    bn2.initialize(1,shape.N6,height,width,bn_weight2);
    relu2.initialize(1,shape.N6,height,width);
}
float* Conv_blk::forward(float *x) {
    float *y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu1.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    y = relu2.forward(y);
    return y;
}
Conv_blk3::Conv_blk3(BLkShape3 shape,unsigned int height,
                     unsigned int width,
                     string conv_weight1,string bn_weight1,string conv_weight2,string bn_weight2,
                     string conv_weight3,string bn_weight3) {
    conv1.initialize(shape.C,shape.N1,shape.N2,shape.N3,height,width," "," "," ", false);
    bn1.initialize(1,shape.N3,height,width,bn_weight1);
    relu1.initialize(1,shape.N3,height,width);
    conv2.initialize(shape.N3,shape.N4,shape.N5,shape.N6,height,width," "," "," ", false);
    bn2.initialize(1,shape.N6,height,width,bn_weight2);
    relu2.initialize(1,shape.N6,height,width);
    conv3.initialize(shape.N6,shape.N7,shape.N8,shape.N9,height,width," "," "," ", false);
    bn3.initialize(1,shape.N6,height,width,bn_weight3);
    relu3.initialize(1,shape.N6,height,width);
}
float * Conv_blk3::forward(float *x) {
    float *y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu1.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    y = relu2.forward(y);
    y = conv3.forward(y);
    y = bn3.forward(y);
    y = relu3.forward(y);
    return y;
}