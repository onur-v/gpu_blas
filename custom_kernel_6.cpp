#include <cuda_runtime.h>
#include <omp.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cblas.h"
#include "custom_kernel_header.hpp"

#if (MREG % BLOCK_SIZE_X != 0) || (MREG % BLOCK_SIZE_Y != 0)
#error "MREG divisibility error"
#endif

// kernel works with specific sizes, so beware
__global__ void __launch_bounds__(BLOCK_SIZE_X *BLOCK_SIZE_Y)
    gemm_kernel_6(realtype *A, realtype *B, realtype *C, int NN, int MM, int KK,
                  realtype alpha, realtype beta, int M_stride_count) {
    __shared__ realtype A_lds_tr[MREG][NLDS * BLOCK_SIZE_Y];
    __shared__ realtype B_lds[MREG][KLDS * BLOCK_SIZE_X];

    realtype result[KLDS][NLDS] = {0.0};

    int N_base = blockIdx.y * BLOCK_SIZE_Y * NLDS;
    int K_base = blockIdx.x * BLOCK_SIZE_X * KLDS;

    for (int i = 0; i < M_stride_count; i++) {
        // SET LDS
        __syncthreads();
        for (int j = 0; j < (MREG / BLOCK_SIZE_X); j++) {
            for (int k = 0; k < NLDS; k++) {
                int lds_row = k * BLOCK_SIZE_Y + threadIdx.y;
                int lds_col = j * BLOCK_SIZE_X + threadIdx.x;
                int A_row = N_base + lds_row;
                int A_col = i * MREG + lds_col;
                int A_ind = A_row * NN + A_col;
                A_lds_tr[lds_col][lds_row] = A[A_ind];
            }
        }
        for (int j = 0; j < (MREG / BLOCK_SIZE_Y); j++) {
            for (int k = 0; k < KLDS; k++) {
                int lds_row = j * BLOCK_SIZE_Y + threadIdx.y;
                int lds_col = k * BLOCK_SIZE_X + threadIdx.x;
                int B_row = i * MREG + lds_row;
                int B_col = K_base + lds_col;
                int B_ind = B_row * KK + B_col;
                B_lds[lds_row][lds_col] = B[B_ind];
            }
        }
        __syncthreads();

        for (int l = 0; l < MREG; l++) {
            for (int j = 0; j < KLDS; j++) {
                for (int k = 0; k < NLDS; k++) {
                    int lds_row = k * BLOCK_SIZE_Y + threadIdx.y;
                    int lds_col = j * BLOCK_SIZE_X + threadIdx.x;
                    result[j][k] += A_lds_tr[l][lds_row] * B_lds[l][lds_col];
                }
            }
        }
    }

    for (int i = 0; i < KLDS; i++) {
        for (int j = 0; j < NLDS; j++) {
            int C_row = N_base + j * BLOCK_SIZE_Y + threadIdx.y;
            int C_col = K_base + i * BLOCK_SIZE_X + threadIdx.x;
            int C_ind = C_row * KK + C_col;
            C[C_ind] = alpha * result[i][j] + beta * C[C_ind];
        }
    }
}

void custom_kernel_6_call(realtype *A, realtype *B, realtype *C, int NN, int MM,
                          int KK, realtype alpha, realtype beta) {
    const int N_stride = NLDS * BLOCK_SIZE_Y;
    const int K_stride = KLDS * BLOCK_SIZE_X;
    int grid_y = (NN + N_stride - 1) / N_stride;
    int grid_x = (KK + K_stride - 1) / K_stride;

    //printf("grid_x = %d, grid_y = %d\n", grid_x, grid_y);

    int M_stride_count = (MM + MREG - 1) / MREG;

    gemm_kernel_6<<<dim3(grid_x, grid_y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(
        A, B, C, NN, MM, KK, alpha, beta, M_stride_count);
}

