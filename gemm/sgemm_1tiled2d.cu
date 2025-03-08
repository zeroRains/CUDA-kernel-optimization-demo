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
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_blocktiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // gridDim : (x,y) (4,4)
    // blockDim : (x) (64) (8x8)
    /* 这里明确一下，这里分为两种规模，一种是全局的计算（全部的A和全部的B计算得到全部的C），一种是局部的计算（一小块A和一小块B计算得到局部的C）
        以256(M)*128(K)， 128(K)* 256(N)的矩阵为例
        grid为(256(M)/64(BN),256(N)/64(BN)) = (4,4)
        block为((BN*BM)/(TN*TM))——>((64*64)/(8*8))
        用人话说，一个block计算一个64*64的子块（全局），一个block里有64个线程，每个线程计算一个8*8的小块（局部）
    */
    

    /*这里是全局的索引*/
    // Block index block的ID
    const uint c_row = blockIdx.y; // 结果矩阵C的第(c_row,c_col)个分块
    const uint c_col = blockIdx.x;

    /*根据全局索引，将A,B,C拉到这个块的起始位置*/
    // Adjust pointers for A, B, and C，A是按行遍历，B是按列遍历，确定起始位置
    A += c_row * BM * K;              // c_row * BM表示绝对行，因为M被分成了若干个BM，当前是第c_row个BM，所以是绝对行，然后K就是A的列，原矩阵A大行的起始位置
    B += c_col * BN;                  // N被分成了若干BN，当前是第c_row个BN，所以绝对列就是c_col * BN，原矩阵B大列的起始位置
    C += c_row * BM * N + c_col * BN; // 行偏移+列偏移，原矩阵C中的对应位置

    /*后续逻辑全是局部的规模*/
    // Thread index within the block C矩阵逻辑 {8*8矩阵}的行列ID
    const uint thread_col = threadIdx.x % (BN / TN); // 列id
    const uint thread_row = threadIdx.x / (BN / TN); // 行id

    // Size of the 2D tile (block tile)
    const uint total_results_block_tile = BM * BN; // 一个block处理的矩阵大小
    // Number of threads needed for a block tile
    const uint number_threads_block_tile = total_results_block_tile / (TM * TN); // 64个8*8的矩阵

    assert(number_threads_block_tile == blockDim.x); // 确保每个线程都能处理一个8*8的矩阵

    // Calculate the shared memory index that this thread is responsible for loading
    // 计算该 Thread 负责加载的共享内存索引绝对
    const uint inner_row_A = threadIdx.x / BK; // A行索引
    const uint inner_col_A = threadIdx.x % BK; // A列索引

    // Calculate the number of rows each thread block loads at a time
    // 计算每个线程块一次加载的行数
    const uint stride_A = number_threads_block_tile / BK; // 8

    const uint inner_row_B = threadIdx.x / BN;            // B行索引
    const uint inner_col_B = threadIdx.x % BN;            // B列索引
    const uint stride_B = number_threads_block_tile / BN; // 1

    // Shared memory for matrix A and B 设置共享内存
    __shared__ float smem_A[BM * BK]; // 64*64
    __shared__ float smem_B[BN * BK]; // 64*64

    // Initialize thread results and register arrays
    float thread_results[TM * TN] = {0.0}; // 结果矩阵 8*8
    float reg_m[TM] = {0.0}; // BA的一列
    float reg_n[TN] = {0.0}; // BB的一行

    // Outer loop
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK) // 计算每个8*8的分区，原始矩阵A选中行（向右），原始矩阵B选中列（向下）
    {
        // Load matrix A and B into shared memory
        for (uint load_offset = 0; load_offset < BM; load_offset += stride_A)
        {
            smem_A[(inner_row_A + load_offset) * BK + inner_col_A] = A[(inner_row_A + load_offset) * K + inner_col_A]; // 这里能这么加是因为上面AB的操作已经将A推到块的起始位置了
        }

        for (uint load_offset = 0; load_offset < BK; load_offset += stride_B)
        {
            smem_B[(inner_row_B + load_offset) * BN + inner_col_B] = B[(inner_row_B + load_offset) * N + inner_col_B]; // 同理
        }

        // Synchronize threads in the block
        __syncthreads();

        // advance the pointers 切换到下一个BK的起始位置
        A += BK;
        B += BK * N;

        // Compute dot product
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) // A小块的列，B小块的行
        {
            // Load matrix A and B into registers取出A一列B一行
            for (uint i = 0; i < TM; i++)
            {
                reg_m[i] = smem_A[(thread_row * TM + i) * BK + dot_idx]; // A小块 行坐标[thrad_row*TM+i]  列坐标[dot_idx]
            }
            for (uint i = 0; i < TN; i++)
            {
                reg_n[i] = smem_B[dot_idx * BN + thread_col * TN + i]; // B小块 行坐标[dot_idx] 列坐标[thread_col * TN + i]
            }
            // Compute multiplication and accumulate results 乘法累加
            for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m)
            {
                for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n)
                { // 结果小块坐标: 行[reg_idx_m] 列[reg_idx_n]
                    thread_results[reg_idx_m * TN + reg_idx_n] +=
                        reg_m[reg_idx_m] * reg_n[reg_idx_n];
                }
            }
        }

        // Synchronize threads in the block
        __syncthreads();
    }

    // Write results back to matrix C 将结果写入最终的矩阵
    for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m)
    {
        for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n)
        {
            C[(thread_row * TM + reg_idx_m) * N + thread_col * TN + reg_idx_n] =
                thread_results[reg_idx_m * TN + reg_idx_n];
        }
    }
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
    int m = 255;
    int n = 255;
    int k = 255;

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