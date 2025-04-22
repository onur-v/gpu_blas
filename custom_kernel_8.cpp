#include <cuda_runtime.h>
#include <omp.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cblas.h"
#include "custom_kernel_header.hpp"

#if (MREG % BLOCK_SIZE_X != 0) || (MREG % BLOCK_SIZE_Y != 0)
//#error "MREG divisibility error"
#endif

#define TN 8
#define TK 4

#if (NLDS % TN != 0) || (KLDS % TK != 0)
#error "TN - TM divisibility error"
#endif

// kernel works with specific sizes, so beware
__global__ void __launch_bounds__(BLOCK_SIZE_X *BLOCK_SIZE_Y)
    gemm_kernel_8(realtype *A, realtype *B, realtype *C, int NN, int MM, int KK,
                  realtype alpha, realtype beta, int M_stride_count) {
    __shared__ realtype A_lds_tr[MREG][NLDS * BLOCK_SIZE_Y];
    __shared__ realtype B_lds[MREG][KLDS * BLOCK_SIZE_X];

    realtype result[KLDS / TK][NLDS / TN][TK][TN] = {0.0};
    realtype A_dbuff[MREG / BLOCK_SIZE_X][NLDS / TN][TN] = {0.0};
    realtype B_dbuff[MREG / BLOCK_SIZE_Y][KLDS / TK][TK] = {0.0};

    int N_base = blockIdx.y * BLOCK_SIZE_Y * NLDS;
    int K_base = blockIdx.x * BLOCK_SIZE_X * KLDS;

    for (int k = 0; k < NLDS; k++) {
        for (int j = 0; j < (MREG / BLOCK_SIZE_X); j++) {
            int lds_row = k * BLOCK_SIZE_Y + threadIdx.y;
            int lds_col = j * BLOCK_SIZE_X + threadIdx.x;
            int A_row = N_base + lds_row;
            int A_col = lds_col;
            int A_ind = A_row * NN + A_col;
            A_lds_tr[lds_col][lds_row] = A[A_ind];
        }
    }
    for (int k = 0; k < KLDS; k++) {
        for (int j = 0; j < (MREG / BLOCK_SIZE_Y); j++) {
            int lds_row = j * BLOCK_SIZE_Y + threadIdx.y;
            int lds_col = k * BLOCK_SIZE_X + threadIdx.x;  //
            int B_row = lds_row;
            int B_col = K_base + lds_col;
            int B_ind = B_row * KK + B_col;
            B_lds[lds_row][lds_col] = B[B_ind];
        }
    }
    __syncthreads();

    for (int i = 1; i <= M_stride_count; i++) {
        // SET LDS
        if (i < M_stride_count) {
            for (int k = 0; k < NLDS / TN; k++) {
                // for (int k = 0; k < NLDS; k++) {
                for (int j = 0; j < (MREG / BLOCK_SIZE_X); j++) {
                    for (int l = 0; l < TN; l++) {
                        int lds_row =
                            k * TN * BLOCK_SIZE_Y + TN * threadIdx.y + l;  //
                        // int lds_row = k * BLOCK_SIZE_Y + threadIdx.y; //
                        int lds_col = j * BLOCK_SIZE_X + threadIdx.x;
                        int A_row = N_base + lds_row;
                        int A_col = i * MREG + lds_col;
                        int A_ind = A_row * NN + A_col;
                        A_dbuff[j][k][l] = A[A_ind];  //
                        // A_dbuff[j][k] = A[A_ind]; //
                    }
                }
            }
            for (int k = 0; k < KLDS / TK; k++) {
                for (int j = 0; j < (MREG / BLOCK_SIZE_Y); j++) {
                    for (int l = 0; l < TK; l++) {
                        int lds_row = j * BLOCK_SIZE_Y + threadIdx.y;
                        int lds_col =
                            k * TK * BLOCK_SIZE_X + TK * threadIdx.x + l;
                        int B_row = i * MREG + lds_row;
                        int B_col = K_base + lds_col;
                        int B_ind = B_row * KK + B_col;
                        B_dbuff[j][k][l] = B[B_ind];  //
                    }
                }
            }
        }

        for (int l = 0; l < MREG; l++) {
            for (int j = 0; j < KLDS / TK; j++) {
                for (int k = 0; k < NLDS / TN; k++) {
                    for (int n = 0; n < TK; n++) {
                        for (int m = 0; m < TN; m++) {
                            int lds_col =
                                j * TK * BLOCK_SIZE_X + TK * threadIdx.x + n;
                            int lds_row =
                                k * TN * BLOCK_SIZE_Y + TN * threadIdx.y + m;
                            result[j][k][n][m] +=
                                A_lds_tr[l][lds_row] * B_lds[l][lds_col];
                        }
                    }
                }
            }
        }
        if (i < M_stride_count) {
            __syncthreads();
            for (int j = 0; j < (MREG / BLOCK_SIZE_X); j++) {
                for (int k = 0; k < NLDS / TN; k++) {
                    // for (int k = 0; k < NLDS; k++) {
                    for (int l = 0; l < TN; l++) {
                        int lds_row =
                            k * TN * BLOCK_SIZE_Y + TN * threadIdx.y + l;
                        // int lds_row = k * BLOCK_SIZE_Y + threadIdx.y; //
                        int lds_col = j * BLOCK_SIZE_X + threadIdx.x;
                        A_lds_tr[lds_col][lds_row] = A_dbuff[j][k][l];  //
                        // A_lds_tr[lds_col][lds_row] = A_dbuff[j][k]; //
                    }
                }
            }
            for (int j = 0; j < (MREG / BLOCK_SIZE_Y); j++) {
                for (int k = 0; k < KLDS / TK; k++) {
                    for (int l = 0; l < TK; l++) {
                        int lds_row = j * BLOCK_SIZE_Y + threadIdx.y;
                        int lds_col =
                            k * TK * BLOCK_SIZE_X + TK * threadIdx.x + l;
                        B_lds[lds_row][lds_col] = B_dbuff[j][k][l];  //
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = 0; i < KLDS / TK; i++) {
        for (int j = 0; j < NLDS / TN; j++) {
            for (int k = 0; k < TK; k++) {
                for (int l = 0; l < TN; l++) {
                    int C_row =
                        N_base + j * TN * BLOCK_SIZE_Y + TN * threadIdx.y + l;
                    int C_col =
                        K_base + i * TK * BLOCK_SIZE_X + TK * threadIdx.x + k;
                    int C_ind = C_row * KK + C_col;
                    C[C_ind] = alpha * result[i][j][k][l] + beta * C[C_ind];
                }
            }
        }
    }
}

void custom_kernel_8_call(realtype *A, realtype *B, realtype *C, int NN, int MM,
                          int KK, realtype alpha, realtype beta) {
    const int N_stride = NLDS * BLOCK_SIZE_Y;
    const int K_stride = KLDS * BLOCK_SIZE_X;
    int grid_y = (NN + N_stride - 1) / N_stride;
    int grid_x = (KK + K_stride - 1) / K_stride;

    // printf("grid_x = %d, grid_y = %d\n", grid_x, grid_y);

    int M_stride_count = (MM + MREG - 1) / MREG;

    gemm_kernel_8<<<dim3(grid_x, grid_y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(
        A, B, C, NN, MM, KK, alpha, beta, M_stride_count);
}

