#hipcc custom_kernel_7.cpp -c -o custom_kernel_7.o -O3  -DRL$1 -Wno-unused -fgpu-flush-denormals-to-zero -ffast-math -fPIE -mcumode #-mwavefrontsize64
hipcc custom_kernel_10.cpp -c -o custom_kernel_10.o -O3  -DRL$1 -Wno-unused -fgpu-flush-denormals-to-zero -ffast-math -fPIE -DKLDS=$5 -DNLDS=$6 -DTN=$7 -DTK=$8 -DSX=$9 -DBLOCK_SIZE_X=$2 -DBLOCK_SIZE_Y=$3 -DMREG=$4
hipcc hipblas_bench.cpp -c -o hipblas_bench.o -O3 -fopenmp -DRL$1 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ -DRL$1 -Wno-unused -fPIE
g++ -o prog_hip hipblas_bench.o custom_kernel_10.o -L/opt/rocm/lib -lhipblas -fopenmp -lamdhip64

