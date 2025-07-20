#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

const uint WARPSIZE = 32;

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

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
// !!!core kernel
template <const int BM,          // 64
          const int BN,          // 64
          const int BK,          // 8
          const int WM,          // 32
          const int WN,          // 32
          const int WMITER,      // 1
          const int WNITER,      // 2
          const int TM,          // 4
          const int TN,          // 4
          const int NUM_THREADS> // 128
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    const int warp_idx = threadIdx.x / WARPSIZE;
    const int warp_row = warp_idx / (BN / WN);
    const int warp_col = warp_idx % (BN / WN);

    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    const int thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);

    const int inner_row_a = threadIdx.x / (BK / 4);
    const int inner_col_a = threadIdx.x % (BK / 4) * 4;

    const int inner_row_b = threadIdx.x / (BN / 4);
    const int inner_col_b = threadIdx.x % (BN / 4) * 4;

    const int stride_a = 4 * NUM_THREADS / BK;
    const int stride_b = 4 * NUM_THREADS / BN;

    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BK * BN];

    A += c_row * BM * K;
    B += c_col * BN;
    C += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN;

    float res[WMITER * WNITER * TN * TM] = {0.0};
    float reg_a[WMITER * TM] = {0.0};
    float reg_b[WNITER * TN] = {0.0};

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        for (int i = 0; i < BM; i += stride_a)
        {
            const float4 tmp = FETCH_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a, K)]);
            smem_a[OFFSET(inner_col_a, i + inner_row_a, BM)] = tmp.x;
            smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = tmp.y;
            smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = tmp.z;
            smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = tmp.w;
        }
        for (int i = 0; i < BK; i += stride_b)
            FETCH_FLOAT4(smem_b[OFFSET(i + inner_row_b, inner_col_b, BN)]) = FETCH_FLOAT4(B[OFFSET(i + inner_row_b, inner_col_b, N)]);

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int idx = 0; idx < BK; idx++)
        {
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx)
                for (int i = 0; i < TM; i += 4)
                    FETCH_FLOAT4(reg_a[warp_sub_row_idx * TM + i]) = FETCH_FLOAT4(smem_a[OFFSET(idx, warp_row * WM + warp_sub_row_idx * WSUBM + thread_row_in_warp * TM + i, BM)]);
            for (int warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx)
                for (int i = 0; i < TN; i += 4)
                    FETCH_FLOAT4(reg_b[warp_sub_col_idx * TN + i]) = FETCH_FLOAT4(smem_b[OFFSET(idx, warp_col * WN + warp_sub_col_idx * WSUBN + thread_col_in_warp * TN + i, BN)]);
            for (int warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; warp_sub_row_idx++)
                for (int warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; warp_sub_col_idx++)
                    for (int m = 0; m < TM; m++)
                        for (int n = 0; n < TN; n++)
                            res[(warp_sub_row_idx * TM + m) * WNITER * TN + warp_sub_col_idx * TN + n] += reg_a[warp_sub_row_idx * TM + m] * reg_b[warp_sub_col_idx * TN + n];
        }
        __syncthreads();
    }
    for (int warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; warp_sub_row_idx++)
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; warp_sub_col_idx++)
        {
            float *c_in = C + (warp_sub_row_idx * WSUBM) * N + warp_sub_col_idx * WSUBN;
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n+=4)
                    FETCH_FLOAT4(c_in[OFFSET(m + thread_row_in_warp * TM, n + thread_col_in_warp * TN, N)]) = FETCH_FLOAT4(res[(warp_sub_row_idx * TM + m) * WNITER * TN + warp_sub_col_idx * TN + n]);
        }
}

void run_sgemm_warp_tiling(float *A, float *B, float *C, int m, int n, int k)
{
    const uint NUM_THREADS = 128;
    const uint BN = 64;
    const uint BM = 64;
    const uint BK = 8;
    const uint WN = 32;
    const uint WM = 32;
    const uint WNITER = 1;
    const uint TN = 4;
    const uint TM = 4;

    dim3 blockDim(NUM_THREADS); // 128

    constexpr uint NUM_WARPS = NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((BN % WN == 0) and (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) ==
                  0);
    constexpr uint WMITER =
        (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    // warpsubtile in warptile
    static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

    static_assert((NUM_THREADS * 4) % BK == 0,
                  "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((NUM_THREADS * 4) % BN == 0,
                  "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(BN % (16 * TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(BM % (16 * TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM)); // 4, 4

    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM,
                            TN, NUM_THREADS>
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
    int n = 256;
    int k = 128;

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C, *d_C_ref;

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

    // Copy data to device
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_C_ref, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C_ref, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_warp_tiling(d_A, d_B, d_C, m, n, k);

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
        run_sgemm_warp_tiling(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);
    return 0;
}