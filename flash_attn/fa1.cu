#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// void atten_naive_cpu(float *Q,
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

void randomize_data(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

void fill_data(float *mat, int N, float value)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = value;
    }
}

// Copy from https://github.com/tspeterkim/flash-attention-minimal
__global__ void flash_attn_v1_kernel(const float *Q,
                                     const float *K,
                                     const float *V,
                                     const int N,
                                     const int d,
                                     const int Tc,
                                     const int Tr,
                                     const int Bc,
                                     const int Br,
                                     const float softmax_scale,
                                     float *l,
                                     float *m,
                                     float *O)
{
    /*
     * grid : (Batch_size, num_heads)
     * block : (Bc)
     */

    int tx = threadIdx.x; // 当前blcok的哪一个threads
    int bx = blockIdx.x;  // batch index
    int by = blockIdx.y;  // head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    // qkv的shape: (bs, nh, N, d)
    // 当前线程对应的全局的qkv坐标 -> (bx * gridDim.y + by) * (N * d)
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    // l是累加结果，m是最大值结果，他们是对d维度的统计，所以比上面就差了一个N-> (bx * gridDim.y + by) * d
    int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

    // Define SRAM for Q,K,V,S 的共享内存 一个threads处理一个分块
    extern __shared__ float sram[];
    // int tile_size = Bc * d;
    const int KV_TILE_SIZE = Bc * d; // size of Kj, Vj
    const int Q_TILE_SIZE = Br * d;  // size of Qi
    // float *Qi = sram;
    // float *Kj = &sram[tile_size];
    // float *Vj = &sram[tile_size * 2];
    // float *S = &sram[tile_size * 3];
    float *Qi = sram;
    float *Kj = &sram[Q_TILE_SIZE];
    float *Vj = &sram[Q_TILE_SIZE + KV_TILE_SIZE];
    float *S = &sram[Q_TILE_SIZE + KV_TILE_SIZE * 2];

    // outer loop KV loop
    for (int j = 0; j < Tc; j++)
    {
        // Load Kj, Vj from HBM to SRAM
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = K[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
        }
        __syncthreads();

        // inner loop Q loop
        for (int i = 0; i < Tr; i++)
        {
            if (tx < Br) // 确保Q的部分只加载Br的长度，至于这里为什么成立，可以看line258
            {
                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++)
                {
                    Qi[(tx * d) + x] = Q[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x];
                }

                float row_m_prev = m[lm_offset + (Br * i) + tx]; // 最大值输出位置
                float row_l_prev = l[lm_offset + (Br * i) + tx]; // 求和输出位置

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++) // 遍历K tile的每一行，Q * K^T -> Q一行和K一行做点积
                {
                    float sum = 0;
                    for (int x = 0; x < d; x++) // 遍历K 一行的每一个元素
                    {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(tx * Bc) + y] = sum; // S的shape为[Br, Bc]

                    if (sum > row_m) // 记录最大值， sum会作为S一行的某一个元素，记录他的最大值，就是记录一行的最大值
                        row_m = sum; // 不同的行是由不同的线程计算的
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0; // 当前行的累加值
                for (int y = 0; y < Bc; y++) // 遍历一行的每一个元素
                {
                    S[(tx * Bc) + y] = __expf(S[(tx * Bc) + y] - row_m);
                    row_l += S[(Bc * tx) + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m); // 更新最大值和累加值
                // 具体一点 row_l = sum(e^{S1-max1}) row_l_prev = sum(e^{S2-max2})
                // 这里max1就是row_m ， max2就是row_m_prev，再令全局最大值g_max = max(max1, max2) = row_m_new
                // row_l_new = e^{max2 - g_max} * row_l_prev + e^{max1 - g_max} * row_l
                //           = e^{max2 - g_max} * sum(e^{S1-max2}) +  e^{max1 - g_max} * sum(e^{S2-max1})
                //           = e^{max2 - g_max} * e^{-max2} * sum(e^{S1}) +  e^{max1 - g_max} * e^{-max1} * sum(e^{S2})
                //           = e^{max2 - g_max - max2} * sum(e^{S1}) +  e^{max1 - g_max - max1} sum(e^{S2})
                //           = e^{-g_max} * sum(e^{S1}) +  e^{-g_max} sum(e^{S2})
                //           = sum(e^{S1} - g_max) + sum(e^{S2} - g_max)
                // 通过迭代更新的方式，得到全局的结果
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);
                // softmax 计算方式 S / row_l
                // Write O, l, m to HBM 
                for (int x = 0; x < d; x++) // 取出V的每一列和S的每一行
                {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; y < Bc; y++)
                    {
                        pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    // 写入全局内存
                    // 前半段(1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new)表示之前的结果修正
                    // 后半段，表示当前的结果的修正
                    O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
        }
        __syncthreads();
    }
}

// Naive CPU implementation of attention
void attn_cpu(float *Q,
              float *K,
              float *V,
              int B,
              int nh,
              int N,
              int D,
              float softmax_scale,
              float *O)
{
    // Iterate over batch size
    for (int b = 0; b < B; ++b)
    {
        // Iterate over number of attention heads
        for (int h = 0; h < nh; ++h)
        {
            // Iterate over query tokens (index i)
            for (int i = 0; i < N; ++i)
            {
                // Allocate memory for attention scores for this query token (shape N)
                float *scores = (float *)malloc(N * sizeof(float));
                if (scores == NULL)
                {
                    fprintf(stderr, "Memory allocation failed\n");
                    return;
                }

                // Calculate attention scores between the current query token and all
                // key tokens (index j)
                for (int j = 0; j < N; ++j)
                {
                    float score = 0.0f;
                    // Calculate dot product over the dimension D (index d)
                    for (int d = 0; d < D; ++d)
                    {
                        score += Q[((b * nh + h) * N + i) * D + d] *
                                 K[((b * nh + h) * N + j) * D + d];
                    }
                    scores[j] = score * softmax_scale; // Use the provided softmax_scale
                }

                // Apply safe softmax
                // Find the maximum score
                float max_score = scores[0];
                for (int j = 1; j < N; ++j)
                {
                    if (scores[j] > max_score)
                    {
                        max_score = scores[j];
                    }
                }

                // Calculate exponentiated values and their sum
                float sum_exp = 0.0f;
                float *weights = (float *)malloc(N * sizeof(float));
                if (weights == NULL)
                {
                    fprintf(stderr, "Memory allocation failed\n");
                    free(scores);
                    return;
                }
                for (int j = 0; j < N; ++j)
                {
                    weights[j] = expf(scores[j] - max_score);
                    sum_exp += weights[j];
                }

                // Normalize to get attention weights
                for (int j = 0; j < N; ++j)
                {
                    weights[j] /= sum_exp;
                }

                // Calculate the weighted sum of value vectors and store in O
                for (int d = 0; d < D; ++d)
                {
                    O[((b * nh + h) * N + i) * D + d] = 0.0f;
                    for (int j = 0; j < N; ++j)
                    {
                        O[((b * nh + h) * N + i) * D + d] +=
                            weights[j] * V[((b * nh + h) * N + j) * D + d];
                    }
                }

                // Free temporary memory
                free(scores);
                free(weights);
            }
        }
    }
}

int main()
{
    // input embedding [B,nh, N, D]
    const int B = 4;   // batch size
    const int nh = 8;  // head number
    const int N = 128; // sequence length
    const int D = 64;  // embedding dimension

    // split kv seq_len to Tc and Q seq_len to Tr
    /* 这里有个有趣的地方：
    FA1论文里：Bc=ceil( sram_max / 4*d), Br=min(ceil( sram_max / 4*d), d)
    这里Br的min十分有灵性，这里可以保证一定会有 Br <= Bc
    看到下面Block_size设置的时候你会发现，Block_size=Bc，
    即使当Br != Bc时，也可以只用简单的if语句就可以完成Br*d的Qi加载, 高!
    s所以设置Bc和Br的时候最好是相等的，可以提高GPU线程的利用率
    */
    const int Bc = 32;                  // KV的分块大小-> Bc * D
    const int Br = 16;                  // Q的分块大小-> Br * D
    const int Tc = ceil((float)N / Bc); // KV的分块数
    const int Tr = ceil((float)N / Br); // Q的分块数

    const float softmax_scale = 1.0 / sqrt(D); // scale值

    // Allocate memory
    float *Q = (float *)malloc(B * nh * N * D * sizeof(float));
    float *K = (float *)malloc(B * nh * N * D * sizeof(float));
    float *V = (float *)malloc(B * nh * N * D * sizeof(float));
    float *O = (float *)malloc(B * nh * N * D * sizeof(float));
    float *O_cpu = (float *)malloc(B * nh * N * D * sizeof(float));
    float *l = (float *)malloc(B * nh * N * sizeof(float));
    float *m = (float *)malloc(B * nh * N * sizeof(float));

    // Initialize data
    randomize_data(Q, B * nh * N * D);
    randomize_data(K, B * nh * N * D);
    randomize_data(V, B * nh * N * D);
    fill_data(O, B * nh * N * D, 0.0f);
    fill_data(l, B * nh * N, 0.0f);
    fill_data(m, B * nh * N, -INFINITY);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc((void **)&d_Q, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_K, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_V, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_O, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_l, B * nh * N * sizeof(float));
    cudaMalloc((void **)&d_m, B * nh * N * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_Q, Q, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, B * nh * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, B * nh * N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate SRAM size needed per block
    const int sram_size =
        (3 * Bc * D * sizeof(float)) + (Bc * Br * sizeof(float)); // 前3个是Q,K,V， 后一个是结果矩阵的一部分，这里指的是一个block使用的共享内存大小
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n",
           max_sram_size,
           sram_size);

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block 每个thread处理一个D的数据，一个block处理Bc*D的数据

    // Launch kernel
    flash_attn_v1_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, D, Tc, Tr, Bc, Br, softmax_scale, d_l, d_m, d_O);

    // Copy result to host
    cudaMemcpy(O, d_O, B * nh * N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Run cpu flash attention
    attn_cpu(Q, K, V, B, nh, N, D, softmax_scale, O_cpu);

    // Check results
    float max_diff = 0.0f;
    for (int i = 0; i < B * nh * N * D; i++)
    {
        max_diff = fmaxf(max_diff, fabsf(O[i] - O_cpu[i]));
    }

    if (max_diff < 0.0001)
    {
        printf("Results are correct! \n");
    }
    else
    {
        printf("Results are incorrect! Max diff: %f\n", max_diff);
    }

    // Free memory
    free_resource(Q, 0);
    free_resource(K, 0);
    free_resource(V, 0);
    free_resource(O, 0);
    free_resource(O_cpu, 0);
    free_resource(l, 0);
    free_resource(m, 0);
    free_resource(d_Q);
    free_resource(d_K);
    free_resource(d_V);
    free_resource(d_O);
    free_resource(d_l);
    free_resource(d_m);

    return 0;
}