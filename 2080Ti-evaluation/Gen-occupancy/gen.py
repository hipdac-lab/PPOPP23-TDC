import subprocess
import codecs
import math
import os,sys
from pathlib import Path
from tqdm import tqdm
import os
try:
    CUDA_PREFIX = os.environ['CUDA_PREFIX']
except:
    print('No CUDA_PREFIX set, set it to /usr/local/cuda/')
    CUDA_PREFIX="/usr/local/cuda/"
def build_switch(H,W):
    R = 3
    S = 3
    Ho = H - (R-1)
    Wo = W - (S-1)
    template = '\t\tcase {}:\n\t\t\t#pragma unroll\n\t\t\tfor ( int r = {}; r < {}; r++) {{\n\t\t\t\t#pragma unroll\n\t\t\t\tfor ( int s = {}; s < {}; s++) {{' \
               '\n\t\t\t\t\tfloat result = v * temp_kernel[r*S+s];\n\t\t\t\t\ttemp_result[({}-r)*{}+({}-s)] += result;\n\t\t\t\t}}\n\t\t\t}}\n\t\tbreak;\n'
    line = '__device__ void switch_function( unsigned int switch_condition,float *temp_kernel,float v,float *temp_result){\n' \
           '\tswitch (switch_condition) {\n'
    for h in range(H):
        for w in range(W):
            r_end = R
            s_end = S
            id = h*W+w
            r_start_condition = (h - Ho + 1)
            r_end_condition = (h+1)
            s_start_condition = (w - Wo + 1)
            s_end_condition = (w+1)
            r_end = min(r_end,r_end_condition)
            r_start = max(0,r_start_condition)
            s_end = min(s_end,s_end_condition)
            s_start = max(0,s_start_condition)
            case_line = template.format(id,r_start,r_end,s_start,s_end,h,(Wo),w)
            line +=case_line
    line += '\n\t}'
    line += '\n}'
    return line
def generate_code(N,C,H,W,TC,TH,TW):
    func_reader = codecs.open('template.cu','r','utf-8')
    switch_func_lines = build_switch(TH+2,TW+2)
    func_lines = func_reader.readlines()
    content = ''
    for line in func_lines:
        content += line
    content = content.replace('#define TH place holder', '#define TH {}'.format(TH))
    content = content.replace('#define TW place holder', '#define TW {}'.format(TW))
    content = content.replace('#define TC place holder', '#define TC {}'.format(TC))
    content = content.replace('#define H place holder', '#define H {}'.format(H))
    content = content.replace('#define W place holder', '#define W {}'.format(W))
    content = content.replace('#define C place holder', '#define C {}'.format(C))
    content = content.replace('#define N place holder', '#define N {}'.format(N))
    #switch_function_place_holder
    content = content.replace('switch_function_place_holder', switch_func_lines)
    writter = codecs.open('temp.cu','w','utf-8')
    writter.write(content)
    writter.close()

if __name__ == '__main__':
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
    ths = [1,2,3,4,5,6,7,8,9,10,11,12]
    tws = [1,2,3,4,5,6,7,8,9,10,11,12]
    tcs = [1,2,4,8,16,32]
    for shape in shape_list:
        c = shape[0]
        n = shape[1]
        h = shape[2]
        w = shape[3]
        blk_dim = ((n - 1)//32 + 1) * 32
        for th in ths:
            if th >= h:
                continue
            for tw in tws:
                if tw>= h:
                    continue
                for tc in tcs:
                    if tc>=c:
                        continue
                    generate_code(n,c,h,w,tc,th,tw)
                    exc_file = './conv'
                    subprocess.run(
                        ["nvcc", "-std=c++11", "-O3", "-I", CUDA_PREFIX + "include", "-L",
                         CUDA_PREFIX + "lib64", "-L", CUDA_PREFIX + "lib", 'temp.cu', "-o",
                         exc_file, "-lcudnn", "-Xcompiler", "-fopenmp" ])
                    subprocess.run([exc_file])