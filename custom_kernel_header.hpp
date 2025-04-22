#ifndef CUSTOM_KERNEL_HEADER_HPP
#define CUSTOM_KERNEL_HEADER_HPP

#ifdef RLF
#define realtype float
#elif defined(RLD)
#define realtype double
#endif

#define WGP_MODE 0
#define OCCUPANCY 4
#define WAVEFRONT 32
#define WAVEPERBLOCK (BLOCK_SIZE / WAVEFRONT)

#define NREG_UQ 2  // 2
#define NREG (NREG_UQ * WAVEPERBLOCK)
#define KREG 2    // 2
#define KINNER 2  // 2

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y)
#define MREG 32
#define KLOOP 1
#define KLDS 16
#define NLDS 8


void custom_kernel_1_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_2_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_3_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_4_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_5_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_6_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_7_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_8_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

void custom_kernel_9_call(realtype *, realtype *, realtype *, int, int, int,
                          realtype, realtype);

#endif
