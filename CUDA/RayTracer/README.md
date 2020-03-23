# GPU Ray Tracer

This is an active project, working towards a physically based, gpu-accelerated path tracer. It is written in pure functional CUDA-C++, using some features from the newly-standardized thrust library (namely, thrust::sort).

## Usage

This project is written and built with separable compilation in mind. Not only can most of the code be run from the host, but the entire renderer is parameterized and can be compiled into a static C++ library that can be linked against to compute accelerated renders on-demand from the host.

## Renders!

As the project gets more mature, I will be populating the output folder with sample renders for your viewing pleasure.

