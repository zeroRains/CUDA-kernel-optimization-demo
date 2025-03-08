#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cmath>
#include <limits>

#ifndef INFINITY
#define INFINITY std::numeric_limits<float>::infinity()
#endif

__global__ void softmax(float *input, float *output, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows)
        return;
    
    float i_max = -INFINITY;
    float sum_exp = 0;
    int pos = idx * cols;
    for (int i = 0; i < cols; i++) // 找最大值
    {
        i_max = i_max < input[pos + i] ? input[pos + i] : i_max;
    }
    for (int i = 0; i < cols; i++) 
    {
        sum_exp += expf(input[pos + i] - i_max);// 计算 e^(x-max)的求和值
    }
    for (int i = 0; i < cols; i++)
    {
        output[pos + i] = expf(input[pos + i] - i_max) / sum_exp; // 计算每个输出 e^(x-max)/sum(e^(x-max))
    }
}

int main()
{
    const size_t rows = 512;
    const size_t cols = 160;
    std::vector<float> input(rows * cols, 0);
    std::vector<float> output(rows * cols, 0);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            input[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }

    float *d_input, *d_output;

    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;

    cudaMemcpy(output.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // free
    cudaFree(d_input);
    cudaFree(d_output);

    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << output[i * cols * j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}