AR=ar
ARFLAGS= -rcv
NVCC=nvcc
CUDAFLAGS= -arch=sm_61 -gencode=arch=compute_61,code=sm_61
CPPFLAGS= -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart -std=c++17 -L. -lbenrt
CC=g++

OBJECTS= vec3.o ray.o render.o camera.o tri.o random.o bvh.o
TEST_FILES= ./tests/test_vec3.cu ./tests/test_ray.cu ./tests/test_camera.cu ./tests/test_tri.cu

default: libbenrt.a

main: main.cpp libbenrt.a
	$(CC) $^ -o $@ $(CPPFLAGS)

test: $(OBJECTS) $(TEST_FILES)
	$(NVCC) $^ -o $@ $(CUDAFLAGS)

device.o: $(OBJECTS)
	$(NVCC) -dlink $(CUDALINKFLAGS) $^ -o device.o

libbenrt.a: $(OBJECTS) device.o
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cu
	$(NVCC) -c $^ -o $@ $(CUDAFLAGS) -dc

.PHONY: clean
clean:
	rm -f *.a *.out *.o ./main ./test

