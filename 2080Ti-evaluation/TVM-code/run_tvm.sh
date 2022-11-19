#!/bin/sh
rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 226 226 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_224_224.txt
time python3 tune-dump.py 1 226 226 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_224_224.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 114 114 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_112_112.txt
time python3 tune-dump.py 1 114 114 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_112_112.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 58 58 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_56_56.txt
time python3 tune-dump.py 1 58 58 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_56_56.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 58 58 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_56_56.txt
time python3 tune-dump.py 1 58 58 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_56_56.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 58 58 64 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_64_56_56.txt
time python3 tune-dump.py 1 58 58 64 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_64_56_56.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 30 30 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_28_28.txt
time python3 tune-dump.py 1 30 30 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_28_28.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 30 30 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_28_28.txt
time python3 tune-dump.py 1 30 30 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_28_28.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 30 30 64 96 3 3 1 0 2000 YOLO8.log 2>&1 | tee 96_64_28_28.txt
time python3 tune-dump.py 1 30 30 64 96 3 3 1 0 2000 YOLO8.log 2>&1 | tee 96_64_28_28.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 30 30 96 160 3 3 1 0 2000 YOLO8.log 2>&1 | tee 160_96_28_28.txt
time python3 tune-dump.py 1 30 30 96 160 3 3 1 0 2000 YOLO8.log 2>&1 | tee 160_96_28_28.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 30 30 96 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_96_28_28.txt
time python3 tune-dump.py 1 30 30 96 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_96_28_28.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 16 16 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_14_14.txt
time python3 tune-dump.py 1 16 16 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_14_14.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 16 16 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_14_14.txt
time python3 tune-dump.py 1 16 16 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_14_14.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 16 16 96 128 3 3 1 0 2000 YOLO8.log 2>&1 | tee 128_96_14_14.txt
time python3 tune-dump.py 1 16 16 96 128 3 3 1 0 2000 YOLO8.log 2>&1 | tee 128_96_14_14.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 16 16 96 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_96_14_14.txt
time python3 tune-dump.py 1 16 16 96 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_96_14_14.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 9 9 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_7_7.txt
time python3 tune-dump.py 1 9 9 32 32 3 3 1 0 2000 YOLO8.log 2>&1 | tee 32_32_7_7.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 9 9 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_7_7.txt
time python3 tune-dump.py 1 9 9 32 64 3 3 1 0 2000 YOLO8.log 2>&1 | tee 64_32_7_7.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 9 9 64 96 3 3 1 0 2000 YOLO8.log 2>&1 | tee 96_64_7_7.txt
time python3 tune-dump.py 1 9 9 64 96 3 3 1 0 2000 YOLO8.log 2>&1 | tee 96_64_7_7.txt

rm YOLO8.log
touch YOLO8.log
time python3 tune.py 1 9 9 160 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_160_7_7.txt
time python3 tune-dump.py 1 9 9 160 192 3 3 1 0 2000 YOLO8.log 2>&1 | tee 192_160_7_7.txt