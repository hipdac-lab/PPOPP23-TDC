import codecs
import os,sys
import subprocess
import time
import shutil
from pathlib import Path
root_dir = os.getcwd()
machine = '2080Ti'
eval_modeling_dir = os.path.join(root_dir,'{}-evaluation/TDC-LAYERS-MODELING'.format(machine))
eval_oracle_dir = os.path.join(root_dir,'{}-evaluation/TDC-LAYERS-ORACLE'.format(machine))
eval_end2end_dir = os.path.join(root_dir,'{}-evaluation/EndToEnd'.format(machine))

#eval_2080Ti_modeling_script = os.path.join(eval_2080Ti_modeling_dir,'compile_run.py')

print('Evaluating performance of TDC core convolution layers oracle...')
time.sleep(1)
os.chdir(eval_oracle_dir)
subprocess.run(["python3", "compile_run.py"])
time.sleep(10)
print('\n\n')


print('Evaluating performance of TDC core convolution layers modeling...')
time.sleep(1)
os.chdir(eval_modeling_dir)
subprocess.run(["python3", "compile_run.py"])
time.sleep(10)
print('\n\n')


print('Evaluating End2END performance with tuned kernels...')
time.sleep(2)
os.chdir(eval_end2end_dir)
subprocess.run(["python3", "build_run.py"])