import codecs
import sys,os
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
def query_best(c,n,h,w, occupancy_file):
    reader = codecs.open(occupancy_file,'r','utf-8')
    lines = reader.readlines()
    target_lines = []
    for line in lines:
        if '{},{},{},{}'.format(c,n,h,w) in line:
            target_lines.append(line)
    t = 10000
    idx = 0
    for index, line in enumerate(target_lines):
        parts = line.split(',')
        run_time = float(parts[-1])
        if run_time < t:
            t = run_time
            idx = index
    return target_lines[idx]
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
    for shape in shape_list:
        c = shape[0]
        n = shape[1]
        h = shape[2]
        w = shape[3]
        line = query_best(c, n, h, w, "a100-occupancy-unrolled")
        parts = line.split(',')
        tc = int(parts[-4])
        tw = int(parts[-5])
        th = int(parts[-6])
        save_code_2_disk(n,c,h,w,tc,th,tw)