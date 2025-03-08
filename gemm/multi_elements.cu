#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <stdlib.h>

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// 一个grid计算一个4*4的小块，BLOCK_SIZE=4
template<int BLOCK_SIZE> __global__ void Multi_Ele(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float * __restrict__ C,
    const int M, // 32
    const int K, // 16
    const int N){// 32
        // 根据cuda特性x是列, y是行
        // blockDim (x,y):(8,8)
        // threadDim (x,y):(4,4)

        float Csub[4] = {0, 0, 0, 0};

        for(int tile_x=0;tile_x<K/blockDim.x;tile_x++){ // K/blockDim.x ->16/4=4
            __shared__ float As[BLOCK_SIZE*BLOCK_SIZE*4]; // 4*4*4
            __shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE*4]; // 4*4*4

            int i = blockDim.y*blockDim.y + threadIdx.y; // 4*4+y 行偏移
            int j = blockDim.x*tile_x + threadIdx.x; // 4*x+x

            // 设置上初始数据
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x*4] = A[i*K+blockDim.x*tile_x+threadIdx.x*4];
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x*4+1] = A[i*K+blockDim.x*tile_x+threadIdx.x*4+1];
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x*4+2] = A[i*K+blockDim.x*tile_x+threadIdx.x*4+2];
            As[threadIdx.y*BLOCK_SIZE+threadIdx.x*4+3] = A[i*K+blockDim.x*tile_x+threadIdx.x*4+3];

            Bs[threadIdx.x*BLOCK_SIZE+threadIdx.y*4] = B[j*N+blockDim.y*blockIdx.y+threadIdx.y*4];
            Bs[threadIdx.x*BLOCK_SIZE+threadIdx.y*4+1] = B[j*N+blockDim.y*blockIdx.y+threadIdx.y*4+1];
            Bs[threadIdx.x*BLOCK_SIZE+threadIdx.y*4+2] = B[j*N+blockDim.y*blockIdx.y+threadIdx.y*4+2];
            Bs[threadIdx.x*BLOCK_SIZE+threadIdx.y*4+3] = B[j*N+blockDim.y*blockIdx.y+threadIdx.y*4+3];

            __syncthreads();

            #pragma unroll
            for(int k=0;k<blockDim.x;k++){ // 矩阵乘法 ，映射关系可能得画图
                Csub[0] += As[threadIdx.y*BLOCK_SIZE+k]*Bs[k*BLOCK_SIZE+threadIdx.x*4];
                Csub[1] += As[threadIdx.y*BLOCK_SIZE+k]*Bs[k*BLOCK_SIZE+threadIdx.x*4+1];
                Csub[2] += As[threadIdx.y*BLOCK_SIZE+k]*Bs[k*BLOCK_SIZE+threadIdx.x*4+2];
                Csub[3] += As[threadIdx.y*BLOCK_SIZE+k]*Bs[k*BLOCK_SIZE+threadIdx.x*4+3];
            }

            __syncthreads();
        }
        // 结果写入矩阵中
        int i = blockIdx.x * blockDim.x + threadIdx.y; 
        C[i*N+blockIdx.y * blockDim.y+threadIdx.x*4] = Csub[0];
        C[i*N+blockIdx.y * blockDim.y+threadIdx.x*4+1] = Csub[1];
        C[i*N+blockIdx.y * blockDim.y+threadIdx.x*4+2] = Csub[2];
        C[i*N+blockIdx.y * blockDim.y+threadIdx.x*4+3] = Csub[3];

    }

const int M = 32;
const int K = 16;
const int N = 32;
const int BLOCK_SIZE = 4;

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, dev); // 获取0号设备的信息
    cudaSetDevice(dev);
    printf("%s starting now ...\n", devprop.name);
    printf("SM数量： %d\n", devprop.multiProcessorCount);
    printf("每个线程块共享内存大小：%fKB\n", devprop.sharedMemPerBlock/1024.0);
    printf("每个线程块最大线程数：%d\n",devprop.maxThreadsPerBlock);
    printf("每个SM最大线程数：%d\n",devprop.maxThreadsPerMultiProcessor);
    printf("每个SM最大线程束数：%d\n",devprop.maxThreadsPerMultiProcessor/32);

    float* A = (float*)malloc(M*K*sizeof(float)); // 32 * 16
    float* B = (float*)malloc(K*N*sizeof(float)); // 16 * 32
    float* cpu_ref = (float*)malloc(M*N*sizeof(float)); // 32 * 32

    float* gpu_A, *gpu_B, *gpu_ref; // 初始化成二维数组
    cudaMalloc((float **)&gpu_A, M*K*sizeof(float)); // 32 * 16
    cudaMalloc((float **)&gpu_B, K*N*sizeof(float)); // 16 * 32
    cudaMalloc((float **)&gpu_ref, M*N*sizeof(float)); // 32 * 32

    for(int i=0;i<M*K;i++){
        A[i] = i%10; // 数字0~9
    }

    for(int i=0;i<K*N;i++){
        B[i] = i; // 数字i
    }

    dim3 grid(M/(BLOCK_SIZE*4), N/(BLOCK_SIZE*4)); // 32/4, 32/4 -> 8, 8
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 4, 4

    double start = seconds();
    cudaMemcpy(gpu_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    Multi_Ele<BLOCK_SIZE><<<grid, block>>>(gpu_A, gpu_B, gpu_ref, M, K, N);
    cudaMemcpy(cpu_ref, gpu_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    double Elpas = seconds() - start;
    printf("using %lf\n", Elpas);

    for(int i=0;i<16;i++){
        printf("%f ", cpu_ref[i]);
    }
    printf("\n");

    free(cpu_ref);
    free(A);
    free(B);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_ref);

    cudaDeviceReset();

    return 0;

}