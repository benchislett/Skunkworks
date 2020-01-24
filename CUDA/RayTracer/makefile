AR=ar
ARFLAGS= -rcv
NVCC=nvcc
CUDAFLAGS= -arch=sm_35
CPPFLAGS= -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart -std=c++17 -L. -lbenrt
CC=g++

OBJECTS= vec3.o ray.o render.o

default: libbenrt.a

main: main.cpp libbenrt.a
	$(CC) $^ -o $@ $(CPPFLAGS)

libbenrt.a: $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cu
	$(NVCC) -c $^ -o $@ $(CUDAFLAGS)

.PHONY: clean
clean:
	rm -f *.a *.o ./main

