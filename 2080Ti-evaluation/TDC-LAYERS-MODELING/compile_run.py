import subprocess
import os
import codecs
from pathlib import Path
try:
    CUDA_PREFIX = os.environ['CUDA_PREFIX']
except:
    print('No CUDA_PREFIX set, set it to /usr/local/cuda/')
    CUDA_PREFIX="/usr/local/cuda/"
cuda_include = os.path.join(CUDA_PREFIX, 'include')
cuda_lib64 = os.path.join(CUDA_PREFIX, 'lib64')
cuda_lib = os.path.join(CUDA_PREFIX, 'lib')
if __name__ == '__main__':
    output_path = '../../evaluation_outcome/2080Ti-layers-eval-modeling.csv'
    if os.path.exists(output_path):
        os.remove(output_path)
    Path(output_path).touch()
    writter = codecs.open(output_path, 'a', 'utf-8')
    writter.write('N,C,H,W,FFT(ms),WinoGradNonFuse(ms),GEMM(ms),TVM(ms),TDC Modeling(ms),Speedup VS FFT,Speedup VS WINO, Speedup VS GEMM, Speedup VS TVM\n')
    writter.close()
    shape_list = [(64,32,224,224),
                  (64,32,112,112),
                  (32,32,56,56),
                  (64,32,56,56),
                  (64,64,56,56),
                  (32,32,28,28),
                  (64,32,28,28),
                  (96,64,28,28),
                  (160,96,28,28),
                  (192,96,28,28),
                  (32,32,14,14),
                  (64,32,14,14),
                  (128,96,14,14),
                  (192,96,14,14),
                  (32,32,7,7),
                  (64,32,7,7),
                  (96,64,7,7),
                  (192,160,7,7)]
    print('HDEADER')
    print('N,C,H,W,FFT(ms),WinoGradNonFuse(ms),GEMM(ms),TVM(ms),TDC Modeling(ms),Speedup VS FFT,Speedup VS WINO, Speedup VS GEMM, Speedup VS TVM')
    for shape in shape_list:
        c = shape[0]
        n = shape[1]
        h = shape[2]
        w = shape[3]
        code_file = '{}_{}_{}_{}.cu'.format(c, n, h, w)
        exc_file = './conv'
        subprocess.run(
            ["nvcc", "-std=c++11", "-O3", "-I", cuda_include, "-L",
             cuda_lib64, "-L", cuda_lib, code_file, "-o",
             exc_file, "-lcudnn", "-Xcompiler", "-fopenmp" ])
        subprocess.run([exc_file])