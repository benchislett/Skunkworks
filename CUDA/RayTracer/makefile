AR=ar
ARFLAGS= -rcv
NVCC=nvcc
CUDAFLAGS= -arch=sm_50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75
CPPFLAGS= -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart -std=c++17 -L. -lbenrt
CC=g++

OBJECTS= vec3.o ray.o render.o
TEST_FILES= ./tests/test_vec3.cpp ./tests/test_ray.cpp

default: libbenrt.a

main: main.cpp libbenrt.a
	$(CC) $^ -o $@ $(CPPFLAGS)

test: $(TEST_FILES) libbenrt.a
	$(CC) $^ -o $@ $(CPPFLAGS)

libbenrt.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cu
	$(NVCC) -c $^ -o $@ $(CUDAFLAGS)

.PHONY: clean
clean:
	rm -f *.a *.out *.o ./main ./test

