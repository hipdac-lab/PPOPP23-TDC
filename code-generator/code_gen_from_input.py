import subprocess
import codecs
import math
import os,sys
from pathlib import Path
from tqdm import tqdm
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
    func_reader = codecs.open('template.template','r','utf-8')
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
    content = content.replace('switch_function_place_holder', switch_func_lines)
    writter = codecs.open('temp.cu','w','utf-8')
    writter.write(content)
    writter.close()
def save_code_2_disk(N,C,H,W,TC,TH,TW):
    func_reader = codecs.open('template.template','r','utf-8')
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
    content = content.replace('switch_function_place_holder', switch_func_lines)
    writter = codecs.open('{}_{}_{}_{}.cu'.format(C,N,H,W),'w','utf-8')
    writter.write(content)
    writter.close()
def simple_tqdm(stars):
    line = ''
    for i in stars:
        line +=i
    print(line)
if __name__ == '__main__':
    ths = [1,2,3,4,5,6,7,8,9,10,11,12]
    tws = [1,2,3,4,5,6,7,8,9,10,11,12]
    tcs = [1,2,4,8,16,32]
    h = input('Enter height: H\n')
    w = input('Enter width: W\n')
    c = input('Enter in channels: C\n')
    n = input('Enter out channels: N\n')
    b = 1
    h = int(h)
    w = int(w)
    c = int(c)
    n = int(n)
    assert b >0
    assert h >0
    assert w >0
    assert c >0
    assert n >0
    print('starting generate kernel code ........')
    blk_dim = ((n - 1)//32 + 1) * 32
    if os.path.exists('tmp.csv'):
        os.remove('tmp.csv')
    Path('tmp.csv').touch()
    generating_process = 0
    iters = 0
    for th in ths:
        if th >= h:
            continue
        for tw in tws:
            if tw>= h:
                continue
            for tc in tcs:
                if tc>=c:
                    continue
                else:
                    iters +=1
    pbar = tqdm(total=iters - 1)
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
                pbar.update(1)
    config_files = codecs.open('tmp.csv','r','utf-8')
    lines = config_files.readlines()
    min_time = 10000000000
    min_index = -1
    for i,line in enumerate(lines):
        segs = line.split(',')
        t = float(segs[-1])
        if t < min_time:
            min_time = t
            min_index = i
    segs = lines[min_index].split(',')
    th = int(segs[0])
    tw = int(segs[1])
    tc = int(segs[2])
    save_code_2_disk(n,c,h,w,tc,th,tw)