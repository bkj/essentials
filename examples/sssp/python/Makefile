VERBOSE=1

# XCOMPILER=$(python3-config --cflags --ldflags --libs)
XPYTHON=-I/home/ubuntu/anaconda3/include/python3.7m -I/home/ubuntu/anaconda3/include/python3.7m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe -isystem /home/ubuntu/anaconda3/include -fdebug-prefix-map=/tmp/build/80754af9/python_1578510683607/work=/usr/local/src/conda/python-3.7.6 -fdebug-prefix-map=/home/ubuntu/anaconda3=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -L/home/ubuntu/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu -L/home/ubuntu/anaconda3/lib -lpython3.7m -lcrypt -lpthread -ldl  -lutil -lrt -lm  -lpython3.7m -lcrypt -lpthread -ldl  -lutil -lrt -lm

cpu:
	c++ -O3 -shared -std=c++14 \
		-I ../../../externals/pybind11/include \
		${XPYTHON}    \
		example.cpp   \
		-o example.so \
		-fPIC

gpu:
	nvcc                                                      \
		-O3                                                     \
		-shared                                                 \
		-std=c++14                                              \
		-Xcompiler "${XPYTHON}"                                 \
		-Xcompiler "-fPIC -march=native"                        \
		-I ../../../externals/pybind11/include                  \
		-I"../.."                                               \
		-I"../../.."                                            \
		-I"../../../externals/moderngpu/src"                    \
		--extended-lambda                                       \
		-c example.cu  \
		-o example.o
		
		g++ -fPIC -march=native example.o -shared -lcudart -L /usr/local/cuda-11.1/lib64 -o example.so
		

clean:
	rm -f *.so