rm -f *.o ; rm -f *.so

swig -python -c++ test.i
mv test_wrap.cxx test_wrap.cu
nvcc  -c test_wrap.cu -Xcompiler "-fPIC -march=native -I /home/ubuntu/anaconda3/include/python3.7m -I /home/ubuntu/anaconda3/pkgs/numpy-1.16.5-py36h95a1406_0/lib/python3.6/site-packages/numpy/core/include" 
g++ -fPIC -march=native test_wrap.o -shared -lcudart -L /usr/local/cuda-11.1/lib64 -o _test.so
rm -f test_wrap.cu test.o test_wrap.o

python -c "import test; print(test.do_testI())"
python -c "import test; print(test.do_testF())"
python -c "import test; print(test.do_another_test())"
python -c "import test; print(test.yet_another_test())"