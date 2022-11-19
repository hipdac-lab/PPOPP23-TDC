# Artifacts of PPoPP'23 paper "TDC: Towards Extremely Efficient CNNs on GPUs via Hardware-Aware Tucker Decomposition"

## Download Artifacts
```
git clone https://github.com/black-cat-sheriff/TDC-PPOPP/
```

## Package Requirements
* cmake(>=3.10)
* cuDNN(>=8)
* CUDA >=11 (A100)
* CUDA >=10 (2080Ti)
* python3(>=3.7)
* tensorly(0.7.4, pip install tensorly==0.7.0)
* timm(0.5.4, pip install timm==0.5.4)
* torch(>=1.4)
* torchvision
* numpy

Note that we require the version of timm to be 0.5.4. 

## 1. SETUP (very important!!!!!!)
```
export CUDA_PREFIX='/usr/local/cuda'
export CUDA_INCLUDE=$CUDA_PREFIX/include
export CUDA_LIB64=$CUDA_PREFIX/lib64
export LD_LIBRARY_PATH=$CUDA_LIB64:$LD_LIBRARY_PATH
export PATH=$CUDA_PREFIX/bin:$PATH
```
Please change `/usr/local/cuda` to the path to your CUDA library. We assume that cuDNN is installed in ``CUDA_PREFIX``, which means that cuDNN header files are in ``CUDA_INCLUDE`` and cuDNN library files are in ``CUDA_LIB64``. Also, please note that some cuDNN versions only have ``lib`` folder rather than ``lib64`` folder, please change ``CUDA_LIB64`` accordingly. 
    
## 2. TDC TRAINED MODEL EVALUATION

### 2.1 Tucker-format model accuracy evaluation.

We provided 6K images to demostrate our models, but you are encouraged to obtain more images from ImageNet.

In our paper, we claim that TDC-resnet18 achieves 69.7% top1 accuracy, TDC-resnet50 has 76.42% top1 accuracy, TDC-densenet121 has 76.3% top1 accuracy, TDC-densenet201 has 76.92% top1 accuracy and TDC-vgg16 has 71.62% top1 accuracy.

```
cd inference
python main.py --model tkc_resnet18 --data-path test_images/
python main.py --model tkc_resnet50 --data-path test_images/
python main.py --model tkc_densenet121 --data-path test_images/
python main.py --model tkc_densenet201 --data-path test_images/
python main.py --model tkc_vgg16 --data-path test_images/
```

Note that `test_images` is the path to the folder where the images used for the demo are saved (i.e. 6K images in total).

You will observe the following result. 

![eval1](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/model-eval.png)

## 2.2 Peformance Evaluation of TDC-generated Core Convolution Layers on 2080Ti

Please go to the main folder and run:
```
python3 run_2080Ti.py
```
It will generate three tables.

### 2.2.1 Comparison among TDC oracle kernel, cuDNN, and TVM. 

In our paper, we claim that our oracle kernel has about 4X speedup compared with cuDNN on average and that our oracle kernel has about 2X speedup compared with TVM on average.

You will get the first table like below, including convolution shapes, convolution schemes, runtimes, and TDC speedups.

![eval2](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/oracle.png)
 
### 2.2.2 Comparison among TDC modeling kernel, cuDNN, and TVM. 

In our paper, we claim that the TDC modeling kernel is around 25% slower than TDC oracle kernel on average.

You will get the second table like below, including convolution shapes, convolution schemes, runtimes, and TDC speedups.

![eval3](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/modeling.png)

### 2.2.3 End-to-end performance comparison among pure cuDNN on original models, pure cuDNN on TK compressed models, and TK compressed models.

In our paper, we claim that (1) TDC oralce and modeling end2end achieve 1.57X/1.45X ~ 3.41X/3.02X speedup compared with original models with pure cuDNN; (2) TDC oralce and modeling end2end achieve 1.04X/0.96X ~ 2.22X/1.95X speedup compared with TK compressed models with pure cuDNN; and (3) TDC oralce and modeling end2end achieve 1.09X/1.05X ~ 1.51X/1.41X speedup compare with TK compressed models with TVM kernels.

You will get the third table like below, each model has two rows - the first one is header and the second one is runtime.

![eval3](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/end2end.png)
   
## 2.3 Performance Evaluation of TDC-generated Core Convolution Layers on A100

Please go to the main folder and run:
```
python3 run_A100.py 
```
It will generate three tables. 

### 2.3.1 Comparison among TDC oracle kernel, cuDNN, and TVM.

In our paper, we claim that our oracle kernel has about 4X speedup compared with cuDNN on average and that our oracle kernel has about 1.7X speedup compared with TVM on average.

You will observe the first table like below, including convolution shapes, convolution schemes, runtimes, and TDC speedups.

![eval2](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/oracle.png)
 
### 2.3.2 Comparison among TDC modeling kernel, cuDNN, and TVM. 

In our paper, we claim that the TDC modeling kernel is around 25% slower than TDC oracle kernel on average. 

You will get the second table like below, including convolution shapes, convolution schemes, runtimes, and TDC speedups.

![eval3](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/modeling.png)
  
### 2.3.3 End-to-end performance comparison among pure cuDNN on original models, pure cuDNN on TK compressed models, and TK compressed models.

In our paper, we claim that (1) TDC oralce and modeling end2end achieve 1.85X/1.71X ~ 4.57X/4.12X speedup compared with original models with pure cuDNN; (2) TDC oralce and modeling end2end achieve 1.27X/1.10X ~ 3.14X/2.86X speedup compare with TK compressed models with pure cuDNN; and (3) TDC oralce and modeling end2end achieve 1.08X/0.96X ~ 1.45X/1.29X speedup compare with TK compressed models with TVM kernels.
   
You will get the third table like below, each model has two rows - the first one is header and the second one is runtime.

![eval3](https://github.com/black-cat-sheriff/TDC-PPOPP/blob/master/images/end2end.png)

## Used Platforms and Dataset

### Platform 1: 
* GPU: Nvidia GTX 2080 Ti (68 SMs, 11 GB)
* OS:  Ubuntu 20.04 LTS
* CUDA: 10.1
* cuDNN: 8.0.4

### Platform 2: 
* GPU: Nvidia Ampere A100(108 SMs, 40GB)
* OS:   Ubuntu 20.04 LTS
* CUDA: 11.0.3
* cuDNN: 8.2.1

### Dataset:
* Imagenet - ILSVRC2012

## External Links
* TVM (https://tvm.apache.org/)
