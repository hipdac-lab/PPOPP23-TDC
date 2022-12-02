import argparse
import json
import sys

import math
import codecs
from collections import OrderedDict

def read_tvm_source(c, n, h, w):
    tvm_code_file = '../TVM-code/{}_{}_{}_{}.cu'.format(c, n, h, w)
    reader = codecs.open(tvm_code_file,'r','utf-8')
    lines = reader.readlines()
    start = 0
    end = 0
    grid = ''
    block = ''
    for index, line in enumerate(lines):
        if 'extern "C" __global__ void' in line:
            start = index
        if 'float check_diff(float *x' in line:
            end = index
        if 'dim3 grid(' in line:
            grid = line
        if 'dim3 block' in line:
            block = line
    source_code = ''
    for i in range(start, end):
        source_code += lines[i]
    return source_code, grid, block

def auto_unroll(H, W):
    out_put = ''
    R = 3
    S = 3
    Ho = H - 2
    Wo = W - 2
    for h in range(H):
        for w in range(W):
            v = 'shared_input[c*(TH+2)*(WPAD) + {} * WPAD + tw_id * TW + {}]'.format(h, w)
            r_end = R
            s_end = S
            r_start_condition = (h - Ho + 1)
            r_end_condition = (h+1)
            s_start_condition = (w - Wo + 1)
            s_end_condition = (w+1)
            r_end = min(r_end,r_end_condition)
            r_start = max(0,r_start_condition)
            s_end = min(s_end,s_end_condition)
            s_start = max(0,s_start_condition)
            for r in range(r_start, r_end):
                for s in range(s_start, s_end):
                    write_location = (h - r) *Wo + w - s
                    #print(h, r, Wo, w, s, write_location)
                    kernel_location = r * 3 + s
                    out_put+='\t\ttemp_result[{}] += {}*{};\n'.format(write_location, v, 'data_array[{}]'.format(kernel_location))
    return out_put
def build_outter_write(TH, TW):
    out = '\t\tswitch(write_w){\n'
    template = '\t\t\tcase {}:\n \t\t\t#pragma unroll\n\t\t\tfor (unsigned int th = 0; th < {}; ++th) {{ \n\t\t\t\t#pragma unroll\n\t\t\t\tfor (unsigned int tw = 0; tw < {}; ++tw) {{ \n\t\t\t\t\tatomicAdd(&outputs[n*H*W+(h_out_start + th) * W+(w_out_start + tw)],temp_result[(th * TW + tw)]);\n\t\t\t\t}}\n\t\t\t}}\n\t\t\tbreak;\n'
    for tw in range(1, TW+1):
        out += template.format(tw, TH, tw)
    out += '\t\t}'
    return out
def build_inner_write(TH, TW):
    header = '__device__ __forceinline__ void switch_write_back(unsigned int write_h, unsigned int write_w, unsigned int h_out_start, unsigned int w_out_start, unsigned int n, float * outputs, float * temp_result){\n'
    out = '\tswitch(write_h){'
    template = '\n\t\tcase {}: \n {} \n\t\tbreak;'
    for th in range(1, TH + 1):
        out += template.format(th, build_outter_write(th, TW))
    out += '\n\t}'
    func_lines = ''
    func_lines += header
    func_lines += out
    func_lines += '\n}'
    return func_lines

def save_code_2_disk(N,C,H,W,TC,TH,TW):
    func_reader = codecs.open('template_benchmark_cudnn.template','r','utf-8')
    switch_func_lines = auto_unroll(TH+2,TW+2)
    func_lines = func_reader.readlines()
    switch_write_lines = build_inner_write(TH, TW)
    content = ''
    for line in func_lines:
        content += line
    tvm_source_code, tvm_grid, tvm_block = read_tvm_source(C, N, H, W)

    content = content.replace('#define TH place holder', '#define TH {}'.format(TH))
    content = content.replace('#define TW place holder', '#define TW {}'.format(TW))
    content = content.replace('#define TC place holder', '#define TC {}'.format(TC))
    content = content.replace('#define H place holder', '#define H {}'.format(H))
    content = content.replace('#define W place holder', '#define W {}'.format(W))
    content = content.replace('#define C place holder', '#define C {}'.format(C))
    content = content.replace('#define N place holder', '#define N {}'.format(N))
    content = content.replace('compute_place_holder', switch_func_lines)
    content = content.replace('switch_write_back_place_holder', switch_write_lines)
    content = content.replace('tvm_source_code_place_holder', tvm_source_code)
    #tvm_grid_place_holder
    #tvm_block_place_holder
    content = content.replace('tvm_grid_place_holder', tvm_grid)
    content = content.replace('tvm_block_place_holder', tvm_block)
    writter = codecs.open('{}_{}_{}_{}.cu'.format(C,N,H,W),'w','utf-8')
    writter.write(content)
    writter.close()
def get_occupancy_dict(lines):
    occupancy_table = {}
    for line in lines:
        parts = line.split(',')
        C = int(parts[0])
        N = int(parts[1])
        H = int(parts[2])
        W = int(parts[3])
        th = int(parts[4])
        tw = int(parts[5])
        tc = int(parts[6])
        occupancy = float(parts[-3])
        K = (C,N,H,W,th,tw,tc)
        occupancy_table[K] = occupancy
    return occupancy_table

def get_result_dict(lines):
    result_dict = {}
    for line in lines:
        parts = line.split(',')
        C = int(parts[0])
        N = int(parts[1])
        H = int(parts[2])
        W = int(parts[3])
        th = int(parts[4])
        tw = int(parts[5])
        tc = int(parts[6])
        occupancy = float(parts[-1])
        K = (C,N,H,W,th,tw,tc)
        result_dict[K] = occupancy
    return result_dict

def compute_latency(C,N,H,W,th,tw,tc,r,s,sms,ths_sm,occupancy_table):
    flops = (th+r-1)*(tw+s-1)*tc*r*s
    bnum = math.ceil(H/th) * math.ceil(C/tc)
    bdim = math.ceil(W/tw) * N
    K = (C,N,H,W,th,tw,tc)
    api_occup = occupancy_table[K]
    bnums_one_wave = math.floor((sms*ths_sm*api_occup)/bdim)
    waves = math.ceil(bnum/bnums_one_wave)
    return flops * waves

def compute_input_vol(C,N,H,W,th,tw,tc):
    in_vol = math.ceil(H/th)  * (th + 2) * (W + 2) * C
    return in_vol
def compute_out_vol(C,N,H,W,th,tw,tc):
    bnum = math.ceil(H/th) * math.ceil(W/tw) * math.ceil(C/tc)
    out_vol = th * tw * N * bnum
    return out_vol
def compute_kernel_vol(C,N,H,W,th,tw,tc):
    bnum = math.ceil(H/th) * math.ceil(W/tw) * math.ceil(C/tc)
    ker_vol = 3 * 3 * tc * N * bnum
    return ker_vol

def build_table(shapes,occupancy_table,num_sm,num_threads):
    ths = [1,2,3,4,5,6,7,8,9,10,11,12]
    tws = [1,2,3,4,5,6,7,8,9,10,11,12]
    #ths = [1,2,3,4,5,6,7]
    #tws = [1,2,3,4,5,6,7]
    tcs = [1,2,4,8,16]
    tiling_table = {}
    for shape in shapes:
        C = shape[0]
        N = shape[1]
        H = shape[2]
        W = shape[3]
        for th in ths:
            for tw in tws:
                for tc in tcs:
                    if th >= H or tw >= W:
                        continue
                    if(C,N,H,W,th,tw,tc) not in occupancy_table.keys():
                        continue
                    K = compute_kernel_vol(C,N,H,W,th,tw,tc)
                    I = compute_input_vol(C,N,H,W,th,tw,tc)
                    O = compute_out_vol(C,N,H,W,th,tw,tc)
                    flops = compute_latency(C,N,H,W,th,tw,tc,3,3,
                                            num_sm,num_threads,occupancy_table)
                    table_key = (C,N,H,W,th,tw,tc)
                    table_value = (K,I,O,flops)
                    tiling_table[table_key] = table_value
    return tiling_table

def oracl(out_file,shapes):
    reader = codecs.open(out_file,'r','utf-8')
    lines = reader.readlines()
    oracl_dict = {}
    for shape in shapes:
        candidates = []
        C = shape[0]
        N = shape[1]
        H = shape[2]
        W = shape[3]
        for line in lines:
            if '{},{},{},{}'.format(C,N,H,W) in line:
                candidates.append(line)
        best = 10000000
        for line in candidates:
            parts = line.split(',')
            t = float(parts[-1])
            if t < best:
                best = t
        oracl_dict[(C,N,H,W)] = best
    return oracl_dict

def modeling(tiling_table, shape, a, b, c, ratio):
    C = shape[0]
    N = shape[1]
    H = shape[2]
    W = shape[3]
    candidates = []
    ths = [1,2,3,4,5,6,7,8,9,10,11,12]
    tws = [1,2,3,4,5,6,7,8,9,10,11,12]
    tcs = [1,2,4,8,16,32]
    for th in ths:
        for tw in tws:
            for tc in tcs:
                if th >= H or tw >= W:
                    continue
                if(C,N,H,W,th,tw,tc) not in tiling_table.keys():
                    continue
                K,I,O,flops = tiling_table[(C,N,H,W,th,tw,tc)]
                data_vol = K*a + I*b + O*c
                candidate = (th,tw,tc,data_vol,flops)
                candidates.append(candidate)
    candidates_filterd_data = candidates[0:math.ceil(len(candidates)*ratio)]
    candidates_filterd_data.sort(key=lambda x: x[-2])
    candidate = candidates_filterd_data[0]
    th = candidate[0]
    tw = candidate[1]
    tc = candidate[2]
    return th, tw, tc

def loss(shape, th, tw, tc, oracl_dict,result_dict):
    C = shape[0]
    N = shape[1]
    H = shape[2]
    W = shape[3]
    oracl = oracl_dict[shape]
    result = result_dict[(C,N,H,W,th,tw,tc)]
    return result/oracl - 1

def print_loss(shape, th, tw, tc, oracl_dict,result_dict):
    C = shape[0]
    N = shape[1]
    H = shape[2]
    W = shape[3]
    oracl = oracl_dict[shape]
    result = result_dict[(C,N,H,W,th,tw,tc)]
    print(shape,result, oracl, result/oracl - 1)
    return result/oracl - 1
if __name__ == '__main__':
    print("starting generate core convolution layers cuda code by analytical modeling on 2080Ti")
    occupancy_table = '2080Ti-occupancy-unrolled'
    shapes = [(64,32,224,224),(64,32,112,112),(32,32,56,56),(64,32,56,56),
              (64,64,56,56),(32,32,28,28),(64,32,28,28),(96,64,28,28),(160,96,28,28),
              (192,96,28,28),(32,32,14,14),(64,32,14,14),
              (128,96,14,14),(192,96,14,14),(32,32,7,7),
              (64,32,7,7),(96,64,7,7),(192,160,7,7)]

    reader = codecs.open(occupancy_table,'r','utf-8')
    num_sm = 68
    num_threads = 1024
    raw_data_lines = reader.readlines()
    occupancy_dict = get_occupancy_dict(raw_data_lines)

    oracl_dict = oracl(occupancy_table,shapes)

    result_dict = get_result_dict(raw_data_lines)

    tiling_table = build_table(shapes,occupancy_dict,num_sm,num_threads)

    avg_loss = 0
    for shape in shapes:
        th, tw, tc = modeling(tiling_table, shape, 0.0, 0.9, 0.1, 0.15)
        avg_loss += print_loss(shape, th, tw, tc, oracl_dict,result_dict)
        c = shape[0]
        n = shape[1]
        h = shape[2]
        w = shape[3]
        save_code_2_disk(n,c,h,w,tc,th,tw)
        print("finished generate Conv kernel code with C = {}, N = {}, H = {}, W = {}".format(c, n, h, w))
    print('average loss is {}'.format(avg_loss/len(shapes)))
    print('generated all Core convolution kernel code in TDC paper, you can evaluate them by run compile_run_layers.py in this directory')
