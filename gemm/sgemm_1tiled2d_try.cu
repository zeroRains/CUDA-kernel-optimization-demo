#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <typename T>
void free_resource(T *p_cpu, T *p_cuda)
{
    if (nullptr != p_cpu)
    {
        delete[] p_cpu;
        p_cpu = nullptr;
    }
    if (nullptr != p_cuda)
    {
        cudaFree(p_cuda);
        p_cuda = nullptr;
    }
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

// Template parameters: -> MxK   KxN  -> MxN
// BM(64), BN(64), BK(8): dimensions of the block
// TM: number of threads per block
// 64, 64, 8, 8, 8 <<<(4,4), (8*8) >>>
// !!!core kernel
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_blocktiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    const int num_per_col = BN / TN;

    const int thread_row = threadIdx.x / num_per_col;
    const int thread_col = threadIdx.x % num_per_col;

    const int total_per_block_tile = BM * BN;
    const int num_per_block = total_per_block_tile / (TN * TM);

    assert(num_per_block == blockDim.x);

    const int inner_row_a = threadIdx.x / BK;
    const int inner_col_a = threadIdx.x % BK;
    const int stride_a = num_per_block / BK;

    const int inner_row_b = threadIdx.x / BN;
    const int inner_col_b = threadIdx.x % BN;
    const int stride_b = num_per_block / BN;

    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BN * BK];

    float res[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        for (int load_offset = 0; load_offset < BM; load_offset += stride_a)
            smem_a[(load_offset + inner_row_a) * BK + inner_col_a] = A[(load_offset + inner_row_a) * K + inner_col_a];
        for (int load_offset = 0; load_offset < BK; load_offset += stride_b)
            smem_b[(load_offset + inner_row_b) * BN + inner_col_b] = B[(load_offset + inner_row_b) * N + inner_col_b];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int idx = 0; idx < BK; idx++)
        {
            for (int i = 0; i < TM; i++)
                reg_m[i] = smem_a[(thread_row * TM + i) * BK + idx];
            for (int i = 0; i < TN; i++)
                reg_n[i] = smem_b[idx * BN + thread_col * TN + i];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    res[m * TN + n] += reg_m[m] * reg_n[n];
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            C[(thread_row * TM + m) * N + thread_col * TN + n] = res[m * TN + n];
}

void run_sgemm_blocktiling_2d(float *A, float *B, float *C, int m, int n, int k)
{
    // m = 256, k = 128, n = 256 -> mxk * kxn = mxn
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    const uint BM = 64;
    const uint BN = 64;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM)); // 256/64=4, 256/64=4 -> 一个block处理64*64的矩阵
    dim3 block_size((BM * BN) / (TM * TN));           // 64-> [ 64/8, 64/8]-> 8,8 -> 一个线程处理8*8的矩阵
    sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
        <<<grid_size, block_size>>>(A, B, C, m, n, k);
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
    int n = 256;
    int k = 128;

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C;

    A = new float[m * k]; // 256 * 128
    B = new float[k * n]; // 128 * 256
    C = new float[m * n]; // 256 * 256
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

    run_sgemm_blocktiling_2d(d_A, d_B, d_C, m, n, k);

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
        run_sgemm_blocktiling_2d(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);

    free_resource(A, d_A);
    free_resource(B, d_B);
    free_resource(C, d_C);
    free_resource(C_ref, (float *)nullptr);

    return 0;
}