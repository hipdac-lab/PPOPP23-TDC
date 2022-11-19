
#include "../inc/common.h"
// conv2_block1
BasicBlock::BasicBlock(unsigned int C1, unsigned int N1, unsigned int N2, unsigned int N3, unsigned int height,
                       unsigned int width){
    conv1.initialize(1,C1,height,width,N1,0,1,1,1," ", false);
    bn1.initialize(1,N1,height,height," ");

    conv2.initialize(1,N1,height,width,N2,1,3,3,1," ", false);
    bn2.initialize(1,N2,height,width," ");

    conv3.initialize(1,N2,height,width,N3,0,1,1,1," ", false);
    bn3.initialize(1,N3,height,width," ");

    relu.initialize(1,N3,height,width);
}

float * BasicBlock::forward(float *x){
    float * y = conv1.forward(x);
    y = bn1.forward(y);
    y = conv2.forward(y);
    y = bn2.forward(y);
    y = conv3.forward(y);
    y = bn3.forward(y);
    y = relu.forward(y);
    return y;
}

BasicBlock_Downsample::BasicBlock_Downsample(unsigned int C1, unsigned int N1, unsigned int N2, unsigned int N3,
                                             unsigned int N4, unsigned int height, unsigned int width,unsigned int stride){
    conv1.initialize(1,C1,height,width,N1,0,1,1,1," ", false);
    bn1.initialize(1,N1,height,height," ");

    conv2.initialize(1,N1,height,width,N2,1,3,3,stride," ", false);
    bn2.initialize(1,N2,height/stride,width/stride," ");

    conv3.initialize(1,N2,height/stride,width/stride,N3,0,1,1,1," ", false);
    bn3.initialize(1,N3,height/stride,width/stride," ");

    relu.initialize(1,N3,height/stride,width/stride);

    conv4.initialize(1,C1,height,width,N4,0,1,1,stride," ", false);
    bn4.initialize(1,N4,height/stride,width/stride," ");

    add.initialize(1,N4,height/stride,width/stride);
}
float * BasicBlock_Downsample::forward(float *x) {
    float *y = conv1.forward(x);
    y = bn1.forward(y);
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