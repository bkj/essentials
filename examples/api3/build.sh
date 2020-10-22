#!/bin/bash

rm -f *.o ; rm -f *.so

swig -python -c++ gunrock_sssp.i
mv gunrock_sssp_wrap.cxx gunrock_sssp_wrap.cu

# <<
nvcc -c gunrock_sssp_wrap.cu \
  -std=c++14 \
  -Xcompiler "-fPIC -march=native -I /home/ubuntu/anaconda3/include/python3.7m -I /home/ubuntu/anaconda3/pkgs/numpy-1.16.5-py36h95a1406_0/lib/python3.6/site-packages/numpy/core/include" \
  -I.. -I../..   -I"../../externals/moderngpu/src" \
  --expt-extended-lambda \
  --expt-relaxed-constexpr 

# # --
# "/usr/local/cuda/bin/nvcc" \
#     -ccbin=g++ \
#     -std=c++14 \
#     -gencode=arch=compute_61,code=\"sm_61,compute_61\"  \
#     --expt-extended-lambda \
#     --expt-relaxed-constexpr \
#     --use_fast_math \
#     --ptxas-options \
#     -v \
#     --relocatable-device-code true \
#     -O3 \
#     --generate-line-info \
#     --compiler-options "-std=c++14 -Wall -Wno-unused-local-typedefs -Wno-strict-aliasing -Wno-unused-function -Wno-format-security -O3" \
#     -c gunrock_sssp_wrap.cu ../../gunrock/util/gitsha1make.c ../../externals/mtx/mmio.cpp  \
#     -I.. -I../..   -I"../../externals/moderngpu/src"  -I"../../externals/rapidjson/include" -I"/usr/local/cuda/bin/../include" -I"../../externals/mtx"  \
#     -Xcompiler -fopenmp \
#     -Xlinker -lgomp  \
#     -Xcompiler -DGUNROCKVERSION=2.0.0 \
#     -Xcompiler "-fPIC -march=native -I /home/ubuntu/anaconda3/include/python3.7m" 
# # >>

g++ -fPIC -march=native gunrock_sssp_wrap.o -shared -lcudart -L /usr/local/cuda-11.1/lib64 -o _gunrock_sssp.so
rm -f gunrock_sssp_wrap.cu gunrock_sssp.o gunrock_sssp_wrap.o

python sssp_gunrock_sssp.py