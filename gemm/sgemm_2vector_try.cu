/*
 * 代码主要参考：https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/blob/master/src/kernel/kernel_6.cuh
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void free_resource(float *ptr, int is_cuda = 1)
{
    if (nullptr != ptr)
    {
        if (is_cuda)
        {
            cudaFree(ptr);
        }
        else
        {
            delete[] ptr;
        }
    }
    ptr = nullptr;
}

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
            {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

// Template parameters:
// BM, BN, BK: dimensions of the block
// TM: number of threads per block
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_vectorize_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    const int num_per_row = BM / TM;
    const int num_per_col = BN / TN;
    const int thread_num = num_per_row * num_per_col;

    const int thread_row = (threadIdx.x / num_per_col) * TM;
    const int thread_col = (threadIdx.x % num_per_col) * TN;

    const int inner_row_a = threadIdx.x / (BK / 4);
    const int inner_col_a = threadIdx.x % (BK / 4) * 4;

    const int inner_row_b = threadIdx.x / (BN / 4);
    const int inner_col_b = threadIdx.x % (BN / 4) * 4;

    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    const int stride_a = BM / ldg_a_num;
    const int stride_b = BK / ldg_b_num;

    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BN * BK];

    float res[TM * TN] = {0.0};
    float ldg_reg_a[4] = {0.0};
    float reg_a[TM] = {0.0};
    float reg_b[TN] = {0.0};

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        for (int i = 0; i < BM; i += stride_a)
        {
            FETCH_FLOAT4(ldg_reg_a[0]) = FETCH_FLOAT4(A[OFFSET(inner_row_a + i, inner_col_a, K)]);
            smem_a[OFFSET(inner_col_a, inner_row_a + i, BM)] = ldg_reg_a[0];
            smem_a[OFFSET(inner_col_a + 1, inner_row_a + i, BM)] = ldg_reg_a[1];
            smem_a[OFFSET(inner_col_a + 2, inner_row_a + i, BM)] = ldg_reg_a[2];
            smem_a[OFFSET(inner_col_a + 3, inner_row_a + i, BM)] = ldg_reg_a[3];
        }
        for (int i = 0; i < BK; i += stride_b)
            FETCH_FLOAT4(smem_b[OFFSET(inner_row_b + i, inner_col_b, BN)]) = FETCH_FLOAT4(B[OFFSET(inner_row_b + i, inner_col_b, N)]);
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int idx = 0; idx < BK; idx++)
        {
            for (int i = 0; i < TM; i += 4)
                FETCH_FLOAT4(reg_a[i]) = FETCH_FLOAT4(smem_a[OFFSET(idx, thread_row + i, BM)]);
            for (int i = 0; i < TN; i += 4)
                FETCH_FLOAT4(reg_b[i]) = FETCH_FLOAT4(smem_b[OFFSET(idx, thread_col + i, BN)]);
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    res[m * TN + n] += reg_a[m] * reg_b[n];
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n += 4)
            FETCH_FLOAT4(C[OFFSET(thread_row + m, thread_col + n, N)]) = FETCH_FLOAT4(res[m * TN + n]);
}

void run_sgemm_vectorize(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    const uint BM = 64;
    const uint BN = 64;

    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(A, B, C, m, n, k);
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

int main(int argc, char *argv[])
{
    int m = 256;
    int n = 128;
    int k = 256;

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    // save reference result
    C_ref = new float[m * n];

    // Initialize matrices
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    // Allocate device memory
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_vectorize(d_A, d_B, d_C, m, n, k);

    // Copy result to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Run reference sgemm
    sgemm_naive_cpu(A, B, C_ref, m, n, k);

    // Verify result
    for (int i = 0; i < m * n; i++)
    {
        if (C[i] != C_ref[i])
        {
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], C[i]);
            return 1;
        }
    }
    printf("Success!\n");

    // Calculate performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        run_sgemm_vectorize(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);

    free_resource(A, 0);
    free_resource(B, 0);
    free_resource(C, 0);
    free_resource(C_ref, 0);

    free_resource(d_A, 1);
    free_resource(d_B, 1);
    free_resource(d_C, 1);

    return 0;
}