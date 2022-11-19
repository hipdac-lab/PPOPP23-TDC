

#include "../inc/common.h"
TenDenseLayer::TenDenseLayer(unsigned int C, unsigned int N1, unsigned int N2, unsigned int H, unsigned int W,
                             string bn1_weight, string conv1_weight, string bn2_weight, string conv2_weight) {
    bn1.initialize(1,C,H,W," ");
    relu1.initialize(1,C,H,W);
    conv1.initialize(1,C,H,W,128,0,1,1,1," ", false);
    bn2.initialize(1,128,H,W," ");
    relu2.initialize(1,128,H,W);
    conv2.initialize(1,128,H,W,32,1,3,3,1," ", true);
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