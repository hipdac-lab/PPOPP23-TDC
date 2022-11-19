
#include "../inc/common.h"
// conv2_block1
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
BasicBlock::BasicBlock(unsigned int C1, unsigned int N1, TkShape shape2, unsigned int N3, unsigned int height,
                       unsigned int width){
    conv1.initialize(1,C1,height,width,N1,0,1,1,1," ", false);
    bn1.initialize(1,N1,height,height," ");

    conv2.initialize(shape2.C,shape2.N1,shape2.N2,shape2.N3,height,width," "," ",
                     " ",false);
    bn2.initialize(1,shape2.N3,height,width," ");

    conv3.initialize(1,shape2.N3,height,width,N3,0,1,1,1," ", false);
    bn3.initialize(1,N3,height,width," ");

    relu.initialize(1,N3,height,width);
}

float * BasicBlock::forward(float *x){
    float * y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    return y;
}

BasicBlock_Downsample::BasicBlock_Downsample(unsigned int C1, unsigned int N1, TkShape shape2, unsigned int N3,
                                             unsigned int N4, unsigned int height, unsigned int width,
                                             unsigned int stride){
    conv1.initialize(1,C1,height,width,N1,0,1,1,1," ", false);
    bn1.initialize(1,N1,height,height," ");

    if(stride == 1){
        conv2.initialize(shape2.C,shape2.N1,shape2.N2,shape2.N3,height,width," "," ",
                         " ",false);
    }else{
        conv2.initialize(shape2.C,shape2.N1,shape2.N2,shape2.N3,height,width," "," ",
                         " ", true);
    }
    bn2.initialize(1,shape2.N3,height/stride,width/stride," ");

    conv3.initialize(1,shape2.N3,height/stride,width/stride,N3,0,1,1,1," ", false);
    bn3.initialize(1,N3,height/stride,width/stride," ");

    relu.initialize(1,N3,height/stride,width/stride);

    conv4.initialize(1,C1,height,width,N4,0,1,1,stride," ", false);
    bn4.initialize(1,N4,height/stride,width/stride," ");

    add.initialize(1,N4,height/stride,width/stride);
}
float * BasicBlock_Downsample::forward(float *x) {
    float *y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    y = conv3.forward(y);
    y = bn3.forward(y);
    y = relu.forward(y);
    float *identity = conv4.forward(x);
    identity = bn4.forward(identity);
    y = add.forward(y,identity);
    return y;
}
