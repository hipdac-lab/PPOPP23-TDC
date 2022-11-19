
#include "../inc/common.h"
// conv2_block1
BasicBlock::BasicBlock(unsigned int C1, unsigned int N1, unsigned int N2, unsigned int height, unsigned int width,
                       string conv1_weight, string conv2_weight, string bn1_weight, string bn2_weight) {
    conv1.initialize(1,C1,height,width,N1,1,3,3,1,conv1_weight, false);
    bn1.initialize(1,N1,height,height,bn1_weight);
    relu.initialize(1,N1,height,width);
    conv2.initialize(1,N1,height,width,N2,1,3,3,1,conv2_weight, false);
    bn2.initialize(1,N2,height,width,bn2_weight);
}

float * BasicBlock::forward(float *x){
    float * y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    return y;
}

BasicBlock_Downsample::BasicBlock_Downsample(unsigned int C1, unsigned int N1, unsigned int N2, unsigned int N3,
                                             unsigned int height, unsigned int width, string conv1_weight,
                                             string conv2_weight, string conv3_weight, string bn1_weight,
                                             string bn2_weight, string bn3_weight) {
    conv1.initialize(1,C1,height,width,N1,2,3,3,2,conv1_weight, false);
    bn1.initialize(1,N1,height/2,width/2,bn1_weight);
    relu.initialize(1,N1,height/2,width/2);
    conv2.initialize(1,N1,height/2,width/2,N2,1,3,3,1,conv2_weight, false);
    bn2.initialize(1,N2,height/2,width/2,bn2_weight);
    conv3.initialize(1,C1,height,width,N3,0,1,1,2,conv3_weight, false);
    bn3.initialize(1,N3,height/2,width/2,bn3_weight);
    add.initialize(1,N3,height/2,width/2);
}
float * BasicBlock_Downsample::forward(float *x) {
    float *y = conv1.forward(x);
    y = bn1.forward(y);
    y = relu.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    float *identity = conv3.forward(x);
    identity = bn3.forward(identity);
    y = add.forward(y,identity);
    return y;
}