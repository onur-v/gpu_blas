#include <cuda_runtime.h>
#include <omp.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cublas_v2.h"

#include "custom_kernel_header.hpp"

int main(int argc, char **argv) {
    cublasHandle_t handle;
    cublasStatus_t stat;

    struct timeval stop1, stop2, start;

    if (argc != 3) {
        printf("Missing/too many inputs!\n");
        return 1;
    }
    int N = atoi(argv[1]);
    int rep_num = atoi(argv[2]);
    // printf("Matrix size is %d and rep. count is %d\n", N, rep_num);

    size_t mat_sz = sizeof(realtype) * N * N;

    realtype *A = (realtype *)malloc(mat_sz);
    realtype *B = (realtype *)malloc(mat_sz);
    realtype *C1 = (realtype *)malloc(mat_sz);
    realtype *C2 = (realtype *)malloc(mat_sz);

    for (int i = 0; i < N * N; i++) {
        A[i] = realtype(rand()) / realtype(RAND_MAX);
        B[i] = realtype(rand()) / realtype(RAND_MAX);
    }

    realtype *A_dev, *B_dev, *C1_dev, *C2_dev;

    cudaMalloc((void **)&A_dev, mat_sz);
    cudaMalloc((void **)&B_dev, mat_sz);
    cudaMalloc((void **)&C1_dev, mat_sz);
    cudaMalloc((void **)&C2_dev, mat_sz);

    stat = cublasCreate(&handle);

    stat = cublasSetMatrix(N, N, sizeof(realtype), A, N, A_dev, N);
    stat = cublasSetMatrix(N, N, sizeof(realtype), B, N, B_dev, N);

    // cudaMemcpy(A_dev, A, mat_sz, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_dev, B, mat_sz, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    const realtype alpha = 1.0, beta = 0.0;

    gettimeofday(&start, NULL);

    for (int i = 0; i < rep_num ; i++) {

#ifdef RLF
        /*
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A,
                    N, B, N, 0.0, C1, N);
        */
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            A_dev, N, B_dev, N, &beta, C1_dev, N);
        //*/
#elif defined(RLD)
        /*
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A,
                    N, B, N, 0.0, C1, N);
        */
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                            A_dev, N, B_dev, N, &beta, C1_dev, N);
        //*/
#endif
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("Operation failure!\n");
    }

    cudaDeviceSynchronize();
    gettimeofday(&stop1, NULL);

    for (int i = 0; i < rep_num; i++) {
        custom_kernel_9_call(A_dev, B_dev, C2_dev, N, N, N, alpha, beta);
    }

    cudaDeviceSynchronize();
    gettimeofday(&stop2, NULL);

    //cudaMemcpy(C1, C1_dev, mat_sz, cudaMemcpyDeviceToHost);
    stat = cublasGetMatrix(N, N, sizeof(realtype), C1_dev, N, C1, N);
    cudaMemcpy(C2, C2_dev, mat_sz, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    realtype norm = 0.0;
    realtype diff_norm = 0.0;
    
    /*
    for (int i = 327; i < 328; i++) {
        for (int j = 0; j < N; j++) {
            if(j % 5 == 0){
                printf("\n");
            }
            printf("%d %d %f %f | ", i, j, C1[i * N + j], C2[i * N + j]);
        }
    }
    */
    
    for (int i = 0; i < N * N; i++) {
        diff_norm += (C1[i] - C2[i]) * (C1[i] - C2[i]);
        norm += C1[i] * C1[i];
    }

    uint64_t microsec_elapsed_1 =
        (stop1.tv_sec - start.tv_sec) * 1000000 + stop1.tv_usec - start.tv_usec;
    uint64_t microsec_elapsed_2 =
        (stop2.tv_sec - stop1.tv_sec) * 1000000 + stop2.tv_usec - stop1.tv_usec;

    double sec_elapsed_1 = (double)microsec_elapsed_1 * 1.0e-6;
    double sec_elapsed_2 = (double)microsec_elapsed_2 * 1.0e-6;
    printf("Total time elapsed for cublas is %f\n", sec_elapsed_1);
    printf("Total time elapsed for custom_blas is %f\n", sec_elapsed_2);
    double gflops_1 = 2.0 * N * N * N * (rep_num ) / (sec_elapsed_1 * 1.0e9);
    double gflops_2 = 2.0 * N * N * N * rep_num / (sec_elapsed_2 * 1.0e9);
    printf("Resulting cublas GFLOPS = %f\n", gflops_1);
    printf("Resulting custom_blas GFLOPS = %f\n", gflops_2);
    printf("Error = %f\n", sqrt(diff_norm / norm));

    return 0;
}
