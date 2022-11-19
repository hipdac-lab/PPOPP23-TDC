
#include "../inc/common.h"
Conv_blk::Conv_blk(unsigned int C, unsigned int N1,unsigned int N2,unsigned int height, unsigned int width,
                   string conv_weight1,string bn_weight1,string conv_weight2, string bn_weight2) {
    conv1.initialize(1,C,height,width,N1,1,3,3,1,conv_weight1, false);
    bn1.initialize(1,N1,height,width,bn_weight1);
    relu1.initialize(1,N1,height,width);
    conv2.initialize(1,N1,height,width,N2,1,3,3,1,conv_weight2, false);
    bn2.initialize(1,N2,height,width,bn_weight2);
    relu2.initialize(1,N2,height,width);
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
Conv_blk3::Conv_blk3(unsigned int C,unsigned int N1,unsigned int N2,unsigned int N3, unsigned int height,
                     unsigned int width,
                     string conv_weight1,string bn_weight1,string conv_weight2,string bn_weight2,
                     string conv_weight3,string bn_weight3) {
    conv1.initialize(1,C,height,width,N1,1,3,3,1,conv_weight1, false);
    bn1.initialize(1,N1,height,width,bn_weight1);
    relu1.initialize(1,N1,height,width);
    conv2.initialize(1,N1,height,width,N2,1,3,3,1,conv_weight2, false);
    bn2.initialize(1,N2,height,width,bn_weight2);
    relu2.initialize(1,N2,height,width);
    conv3.initialize(1,N2,height,width,N3,1,3,3,1,conv_weight3, false);
    bn3.initialize(1,N3,height,width,bn_weight3);
    relu3.initialize(1,N3,height,width);
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