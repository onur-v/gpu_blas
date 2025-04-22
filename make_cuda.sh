nvcc custom_kernel_6.cpp -c -o custom_kernel_6.o -O3  -DRL$1 -ftz=true -use_fast_math -gencode=arch=compute_89,code=sm_89 -x cu
nvcc custom_kernel_9.cpp -c -o custom_kernel_9.o -O3  -DRL$1 -ftz=true -use_fast_math -gencode=arch=compute_89,code=sm_89 -x cu
nvcc cublas_bench.cpp -c -o cublas_bench.o -O3 -Xcompiler -fopenmp -DRL$1 -DRL$1 -Xcompiler -Wno-unused
nvcc -o prog_cuda cublas_bench.o custom_kernel_6.o custom_kernel_9.o -L/usr/local/cuda/lib64 -lcublas -lopenblas -Xcompiler -fopenmp

