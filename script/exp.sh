#!/bin/bash

#Benchmark
out_k=./k-as-x.out.a100
out_n=./n-as-x.out.a100

python3 k-as-x.py -i ${out_k} -o ./k-as-x-a100.png
python3 n-as-x.py -i ${out_n} -o ./n-as-x-a100.png
python3 speedup.py -k ${out_k} -n ${out_n} -o speedup.csv
