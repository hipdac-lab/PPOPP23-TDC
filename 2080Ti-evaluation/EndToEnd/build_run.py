import codecs
import os,sys
import subprocess
import time
import shutil
from pathlib import Path
root_dir = os.getcwd()
eval_dir = Path(root_dir).parent.absolute()
eval_dir = Path(eval_dir).parent.absolute()
eval_dir = os.path.join(eval_dir, 'evaluation_outcome')
machine = '2080Ti'
print("starting build")
build_list = []
build_list.append(os.path.join(root_dir,'densenet121-original/build'))
build_list.append(os.path.join(root_dir,'densenet121-tk/build'))

build_list.append(os.path.join(root_dir,'densenet201-original/build'))
build_list.append(os.path.join(root_dir,'densenet201-tk/build'))

build_list.append(os.path.join(root_dir,'resnet18-original/build'))
build_list.append(os.path.join(root_dir,'resnet18-tk/build'))

build_list.append(os.path.join(root_dir,'resnet50-original/build'))
build_list.append(os.path.join(root_dir,'resnet50-tk/build'))

build_list.append(os.path.join(root_dir,'vgg16-original/build'))
build_list.append(os.path.join(root_dir,'vgg16-tk/build'))

for build_dir in build_list:
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    os.chdir(build_dir)
    subprocess.run(["cmake",".."])
    subprocess.run(["make","-j"])

print("end building")
time.sleep(2)


densenet121_conv_dict={(32,32,56,56):6, (32,32,28,28):12, (32,32,14,14):24, (32,32,7,7):16}
densenet201_conv_dict={(64,32,56,56):6, (64,32,28,28):12, (64,32,14,14):48, (64,32,7,7):32}
resnet18_conv_dict={(64,32,56,56):1, (32,32,56,56):3, (96,64,28,28):3, (128,96,14,14):3, (192,160,7,7):3}
resnet50_conv_dict={(64,32,56,56):1, (32,32,56,56):2, (64,32,28,28):3, (64,32,14,14):5, (96,64,7,7):2}
vgg16_conv_dict={(64,32,224,224):1, (64,32,112,112):2, (64,32,56,56):2, (64,64,56,56):1,
                 (160,96,28,28):2, (192,96,28,28):1, (192,96,14,14):3}

def compute_saving(dict1, dict2, dict3):
    total = 0.0
    for shape in dict1.keys():
        nums = dict1[shape]
        total += (dict2[shape] - dict3[shape]) * nums
    return total
def query_layers_performance():
    modeling_dict = {}
    gemm_dict = {}
    oracle_dict = {}
    tvm_dict = {}
    modeling_output = os.path.join(eval_dir, '{}-layers-eval-modeling.csv'.format(machine))
    oracle_output = os.path.join(eval_dir, '{}-layers-eval-oracle.csv'.format(machine))
    reader = codecs.open(modeling_output,'r','utf-8')
    lines = reader.readlines()
    for line in lines[1:]:
        parts = line.split(',')
        N = int(parts[0])
        C = int(parts[1])
        H = int(parts[2])
        W = int(parts[3])
        gemm = float(parts[6])
        tvm = float(parts[7])
        tdc = float(parts[8])
        gemm_dict[(C,N,H,W)] = gemm
        modeling_dict[(C,N,H,W)] = tdc
        tvm_dict[(C,N,H,W)] = tvm
    reader.close()
    reader = codecs.open(oracle_output,'r','utf-8')
    lines = reader.readlines()
    for line in lines[1:]:
        parts = line.split(',')
        N = int(parts[0])
        C = int(parts[1])
        H = int(parts[2])
        W = int(parts[3])
        tdc = float(parts[8])
        oracle_dict[(C,N,H,W)] = tdc
    reader.close()
    return gemm_dict, oracle_dict, modeling_dict, tvm_dict

def query_performance():
    gemm_dict, oracle_dict, modeling_dict, tvm_dict = query_layers_performance()
    result_dict = {}
    result_output = os.path.join(eval_dir, '{}-end2end.csv'.format(machine))
    reader = codecs.open(result_output,'r','utf-8')
    lines = reader.readlines()
    for line in lines:
        parts = line.split(',')
        model_name = parts[0]
        p = float(parts[1])
        result_dict[model_name] = p
    keys = []
    for key in result_dict.keys():
        keys.append(key)
    for key in keys:
        if key == 'densenet121-tk':
            conv_dict = densenet121_conv_dict
            tdc_model_p = result_dict[key]
            tdc_oracle_p = result_dict[key]
            tvm_p = result_dict[key]
            modeling_saving = compute_saving(conv_dict, gemm_dict, modeling_dict)
            oracle_saving = compute_saving(conv_dict, gemm_dict, oracle_dict)
            tvm_saving = compute_saving(conv_dict, gemm_dict, tvm_dict)
            tdc_model_p -= modeling_saving
            tdc_oracle_p -= oracle_saving
            tvm_p -= tvm_saving
            result_dict['densenet121-tk-modeling'] = tdc_model_p
            result_dict['densenet121-tk-oracle'] = tdc_oracle_p
            result_dict['densenet121-tk-tvm'] = tvm_p
        elif key == 'densenet201-tk':
            conv_dict = densenet201_conv_dict
            tdc_model_p = result_dict[key]
            tdc_oracle_p = result_dict[key]
            tvm_p = result_dict[key]
            modeling_saving = compute_saving(conv_dict, gemm_dict, modeling_dict)
            oracle_saving = compute_saving(conv_dict, gemm_dict, oracle_dict)
            tvm_saving = compute_saving(conv_dict, gemm_dict, tvm_dict)
            tdc_model_p -= modeling_saving
            tdc_oracle_p -= oracle_saving
            tvm_p -= tvm_saving
            result_dict['densenet201-tk-modeling'] = tdc_model_p
            result_dict['densenet201-tk-oracle'] = tdc_oracle_p
            result_dict['densenet201-tk-tvm'] = tvm_p
        elif key == 'resnet18-tk':
            conv_dict = resnet18_conv_dict
            tdc_model_p = result_dict[key]
            tdc_oracle_p = result_dict[key]
            tvm_p = result_dict[key]
            modeling_saving = compute_saving(conv_dict, gemm_dict, modeling_dict)
            oracle_saving = compute_saving(conv_dict, gemm_dict, oracle_dict)
            tvm_saving = compute_saving(conv_dict, gemm_dict, tvm_dict)
            tdc_model_p -= modeling_saving
            tdc_oracle_p -= oracle_saving
            tvm_p -= tvm_saving
            result_dict['resnet18-tk-modeling'] = tdc_model_p
            result_dict['resnet18-tk-oracle'] = tdc_oracle_p
            result_dict['resnet18-tk-tvm'] = tvm_p
        elif key == 'resnet50-tk':
            conv_dict = resnet50_conv_dict
            tdc_model_p = result_dict[key]
            tdc_oracle_p = result_dict[key]
            tvm_p = result_dict[key]
            modeling_saving = compute_saving(conv_dict, gemm_dict, modeling_dict)
            oracle_saving = compute_saving(conv_dict, gemm_dict, oracle_dict)
            tvm_saving = compute_saving(conv_dict, gemm_dict, tvm_dict)
            tdc_model_p -= modeling_saving
            tdc_oracle_p -= oracle_saving
            tvm_p -= tvm_saving
            result_dict['resnet50-tk-modeling'] = tdc_model_p
            result_dict['resnet50-tk-oracle'] = tdc_oracle_p
            result_dict['resnet50-tk-tvm'] = tvm_p
        else:
            conv_dict = vgg16_conv_dict
            tdc_model_p = result_dict[key]
            tdc_oracle_p = result_dict[key]
            tvm_p = result_dict[key]
            modeling_saving = compute_saving(conv_dict, gemm_dict, modeling_dict)
            oracle_saving = compute_saving(conv_dict, gemm_dict, oracle_dict)
            tvm_saving = compute_saving(conv_dict, gemm_dict, tvm_dict)
            tdc_model_p -= modeling_saving
            tdc_oracle_p -= oracle_saving
            tvm_p -= tvm_saving
            result_dict['vgg16-tk-modeling'] = tdc_model_p
            result_dict['vgg16-tk-oracle'] = tdc_oracle_p
            result_dict['vgg16-tk-tvm'] = tvm_p
    d121_header_list = ['densenet121','densenet121-tk','densenet121-tk-tvm','densenet121-tk-modeling','densenet121-tk-oracle']
    d201_header_list = ['densenet201','densenet201-tk','densenet201-tk-tvm','densenet201-tk-modeling','densenet201-tk-oracle']
    r18_header_list = ['resnet18','resnet18-tk','resnet18-tk-tvm','resnet18-tk-modeling','resnet18-tk-oracle']
    r50_header_list = ['resnet50','resnet50-tk','resnet50-tk-tvm','resnet50-tk-modeling','resnet50-tk-oracle']
    v16_header_list = ['vgg16','vgg16-tk','vgg16-tk-tvm','vgg16-tk-modeling','vgg16-tk-oracle']
    header = ''
    print('printing result...')
    time.sleep(1)
    for key in d121_header_list:
        header += '{}(Run time),'.format(key)
    header = header[:-1]
    output = ''
    for key in d121_header_list:
        output += '{}(ms),'.format(result_dict[key])
    output = output[:-1]
    print(header)
    print(output)

    header = ''
    for key in d201_header_list:
        header += '{}(Run time),'.format(key)
    header = header[:-1]
    output = ''
    for key in d201_header_list:
        output += '{}(ms),'.format(result_dict[key])
    output = output[:-1]
    print(header)
    print(output)

    header = ''
    for key in r18_header_list:
        header += '{}(Run time),'.format(key)
    header = header[:-1]
    output = ''
    for key in r18_header_list:
        output += '{}(ms),'.format(result_dict[key])
    output = output[:-1]
    print(header)
    print(output)

    header = ''
    for key in r50_header_list:
        header += '{}(Run time),'.format(key)
    header = header[:-1]
    output = ''
    for key in r50_header_list:
        output += '{}(ms),'.format(result_dict[key])
    output = output[:-1]
    print(header)
    print(output)

    header = ''
    for key in v16_header_list:
        header += '{}(Run time),'.format(key)
    header = header[:-1]
    output = ''
    for key in v16_header_list:
        output += '{}(ms),'.format(result_dict[key])
    output = output[:-1]
    print(header)
    print(output)



time.sleep(2)
os.chdir(eval_dir)
if os.path.exists('{}-end2end.csv'.format(machine)):
    os.remove('{}-end2end.csv'.format(machine))
Path('{}-end2end.csv'.format(machine)).touch()
print('Running......')
for build_dir in build_list:
    os.chdir(build_dir)
    subprocess.run(["./test"])
query_performance()


