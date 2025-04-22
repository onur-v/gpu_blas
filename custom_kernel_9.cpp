#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
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

//#define TN 8
//#define TK 4
//#define SX 1

#define LDS_ARR_LENGTH (MREG * (NLDS * BLOCK_SIZE_Y + KLDS * BLOCK_SIZE_X))
#ifdef RLF
#define LDS_SZ LDS_ARR_LENGTH * 4
#elif defined(RLD)
#define LDS_SZ LDS_ARR_LENGTH * 8
#endif

#if LDS_SZ > 65536
#error "Not enough LDS space!"
#endif

#if (SX * BLOCK_SIZE > NLDS * MREG * BLOCK_SIZE_Y)
#error "Change SX or NLDS!"
#endif

#if ((MREG * NLDS) / (BLOCK_SIZE_X * SX)) / 4 == 0
#define SY ((MREG * NLDS) / (BLOCK_SIZE_X * SX)) % 4
#define GY 1
#else
#define SY 4
#define GY ((MREG * NLDS) / (BLOCK_SIZE_X * SX)) / 4
#endif
#define TX (MREG / SX)
#define TY (BLOCK_SIZE / TX)

#if (NLDS % TN != 0) || (KLDS % TK != 0)
#error "TN - TM divisibility error"
#endif

// kernel works with specific sizes, so beware
__global__ void __launch_bounds__(BLOCK_SIZE_X *BLOCK_SIZE_Y)
    gemm_kernel_9(realtype *A, realtype *B, realtype *C, int NN, int MM, int KK,
                  realtype alpha, realtype beta, int M_stride_count) {
    __shared__ realtype A_lds_tr[MREG][NLDS * BLOCK_SIZE_Y];
    //__shared__ realtype Pad[4];
    __shared__ realtype B_lds[MREG][KLDS * BLOCK_SIZE_X];

    realtype result[KLDS / TK][NLDS / TN][TK][TN] = {0.0};
    // realtype C_temp[KLDS / TK][NLDS / TN][TK][TN];
    //  realtype A_dbuff[MREG / BLOCK_SIZE_X][NLDS / TN][TN] = {0.0};
    realtype A_dbuff[GY][SY][SX];
    realtype B_dbuff[MREG / BLOCK_SIZE_Y][KLDS / TK][TK];

    int N_base = blockIdx.y * BLOCK_SIZE_Y * NLDS;
    int K_base = blockIdx.x * BLOCK_SIZE_X * KLDS;
    int flatIdx = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
    int A_idx_x = flatIdx / TY;
    int A_idx_y = flatIdx % TY;

    for (int j = 0; j < GY; j++) {
        for (int k = 0; k < SY; k++) {
            for (int l = 0; l < SX; l++) {
                int lds_row = j * SY * TY + SY * A_idx_y + k;
                int lds_col = SX * A_idx_x;
                int A_row = N_base + lds_row;
                int A_col = lds_col;
                int A_ind = A_row * NN + A_col;
                A_lds_tr[lds_col + l][lds_row] = A[A_ind + l];
            }
        }
    }

    for (int k = 0; k < KLDS / TK; k++) {
        for (int j = 0; j < (MREG / BLOCK_SIZE_Y); j++) {
            for (int l = 0; l < TK; l++) {
                int lds_row = j * BLOCK_SIZE_Y + threadIdx.y;
                int lds_col = k * TK * BLOCK_SIZE_X + TK * threadIdx.x + l;
                int B_col = K_base + lds_col;
                int B_ind = lds_row * KK + B_col;
                B_lds[lds_row][lds_col] = B[B_ind];
            }
        }
    }

    __syncthreads();

    for (int i = 1; i <= M_stride_count; i++) {
        // SET LDS
        if (i < M_stride_count) {
            for (int j = 0; j < GY; j++) {
                for (int k = 0; k < SY; k++) {
                    for (int l = 0; l < SX; l++) {
                        int row = j * SY * TY + SY * A_idx_y + k;
                        int col = SX * A_idx_x + l;
                        int A_row = N_base + row;
                        int A_col = i * MREG + col;
                        int A_ind = A_row * NN + A_col;
                        A_dbuff[j][k][l] = A[A_ind];
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
            /*
            int last = i;
            if(last < (KLDS / TK)*(NLDS)){
                int m = last % (KLDS / TK);
                int j_l = last / (KLDS / TK);
                int j = j_l / TN;
                int l = j_l % TN;
                for (int k = 0; k < TK; k++) {
                    int C_row =
                        N_base + j * TN * BLOCK_SIZE_Y + TN * threadIdx.y + l;
                    int C_col =
                        K_base + m * TK * BLOCK_SIZE_X + TK * threadIdx.x + k;
                    int C_ind = C_row * KK + C_col;
                    C_temp[m][j][k][l] = C[C_ind];
                }
            }
            */
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
            ///*
            for (int j = 0; j < GY; j++) {
                for (int l = 0; l < SX; l++) {
                    for (int k = 0; k < SY; k++) {
                        int row = j * SY * TY + SY * A_idx_y;
                        int col = SX * A_idx_x + l;
                        A_lds_tr[col][row + k] = A_dbuff[j][k][l];
                    }
                }
            }
            //*/
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

void custom_kernel_9_call(realtype *A, realtype *B, realtype *C, int NN, int MM,
                          int KK, realtype alpha, realtype beta) {
    const int N_stride = NLDS * BLOCK_SIZE_Y;
    const int K_stride = KLDS * BLOCK_SIZE_X;
    int grid_y = (NN + N_stride - 1) / N_stride;
    int grid_x = (KK + K_stride - 1) / K_stride;

    // printf("grid_x = %d, grid_y = %d\n", grid_x, grid_y);
    static int once = 1;
    if (once) {
        printf("SX = %d \n", SX);
        printf("SY = %d \n", SY);
        printf("TX = %d \n", TX);
        printf("TY = %d \n", TY);
        printf("GY = %d \n", GY);
        once = 0;
    }

    int M_stride_count = (MM + MREG - 1) / MREG;

    gemm_kernel_9<<<dim3(grid_x, grid_y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(
        A, B, C, NN, MM, KK, alpha, beta, M_stride_count);
}
