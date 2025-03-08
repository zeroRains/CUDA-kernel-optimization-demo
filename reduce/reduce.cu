#include <iostream>
#define THREAD_PER_BLOCK 128
#include <time.h>
#include <sys/time.h>

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void reduce1(int *d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK]; // data[128]
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x; // 获取全局位置
    unsigned int tid = threadIdx.x; // 一个grid，一个grid地计算
    // 这个不是覆盖了吗？ 覆盖不了每个block的共享内存是隔离的，也就是一个block内才能使用共享内存
    sdata[tid] = d_in[idx]; 
    // 上面这步，就是把一个block的128条数据全部加载到共享内存
    __syncthreads();

    for(unsigned int i=1;i<blockDim.x;i*=2){ // blockDim.x是128，gridDim.x是4
        if(tid%(2*i)==0){
            sdata[tid] = sdata[tid] + sdata[tid+i]; // reduce是相加
        }
        __syncthreads();
    }
    /* 上面的for循环给大家模拟一下，假设一个block只有8个线程
    编号： 0,   1, 2,      3, 4, 5,     6, 7
    i=1：  √       √          √         √
           0+1 ,   2+3,       4+5,      6+7
           1       5          9         13
    i=2:   √                  √
           1+5                9+13
           6                  22
    i=3:   √
           6+22
           28
    */
    // 然后将结果写入到对应的block中
    if(tid==0){
        d_out[blockIdx.x] = sdata[tid];
    }
}


__global__ void reduce2(int* d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int tid = threadIdx.x; //0~127

    sdata[tid] = d_in[idx];
    __syncthreads(); // 和上面一样加载内存
    // 下面的计算原理好像也和上面一样
    for(int i=1;i<blockDim.x;i*=2){ // 128
        int index = 2*i*tid;//乘2因为要累加 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, ...
        if(index<blockDim.x){
            sdata[index] = sdata[index] + sdata[index+i]; 
        }
        __syncthreads();
    }
        /* 上面的for循环给大家模拟一下，假设一个block只有8个线程
    编号： 0,   1, 2,      3, 4, 5,     6, 7
    i=1：  √       √          √         √
           0+1 ,   2+3,       4+5,      6+7
           1       5          9         13
    i=2:   √                  √
           1+5                9+13
           6                  22
    i=3:   √
           6+22
           28
    */
    // 然后将结果写入到对应的block中
    if(tid==0) d_out[blockIdx.x] = sdata[tid];
}

__global__ void reduce3(int* d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = d_in[idx];
    __syncthreads(); // 同上

    for(int s=blockDim.x/2;s>0;s>>=1){ 
        if(tid<s){ // 用原本一半的线程处理 0,1,2,3 （如果总数是8，就只有这4个编号的线程在干活）
            sdata[tid] = sdata[tid] + sdata[tid+s];
        }   //解决bank冲突
        /*
        为了提高内存读写带宽，共享内存被分割成了32个等大小的内存块，即Bank
        因为一个Warp有32个线程，相当于一个线程对应一个内存Bank。
        Bank Conflict是指在同一时间内，一个warp中的多个线程尝试访问共享内存中同一个bank的不同地址时发生的冲突。

        上述修改后，每个线程访问的地址不再相邻，而是相隔一个bank，这样就避免了bank conflict。
        下面例子看不出来，因为总数据量太小，但是如果数据量大，就会出现bank conflict
    编号： 0,      1,         2,        3, 4, 5,     6, 7
    i=1：  √       √          √         √
           0+4 ,   1+5,       2+6,      3+7
           4       6          8         10
    i=2:   √       √
           4+6     8+10
           10      18
    i=3:   √
           10+18
           28
        */
        __syncthreads();
    }

    if(tid==0) d_out[blockIdx.x] = sdata[tid];

}

__global__ void reduce4(int *d_in,int *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];  // 从全局内存中多添加一个元素
    __syncthreads();
    //闲置线程先做一次加法，防止线程利用率低下

    // do reduction in shared mem 这个就是reduce3防止bank conflict的写法
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    /* 上面的for循环给大家模拟一下，假设一个block只有4个线程
    数据： 0,      1,         2,        3, 4, 5,     6, 7
    编号： 0       1          2         3 
    加载： 0+4     1+5        2+6       3+7
           4       6          8        10
    i=1：  √       √          
           4+8,    6+10,
           12      16   
    i=2:   √           
           12+16         
           28           
    */
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

// volatile确保每次访问都是最新的, warp有32个线程
__device__ void warpReduce(volatile int* cache, int tid){ // warp内做reduce, 下面的数字是索引
    cache[tid] += cache[tid+16]; // 0+16, 1+17, 2+18, 3+19, 4+20, 5+21, 6+22, 7+23
    __syncwarp();
    cache[tid] += cache[tid+8];  // 0+8, 1+9, 2+10, 3+11, 4+12, 5+13, 6+14, 7+15
    __syncwarp();
    cache[tid] += cache[tid+4];  // 0+4, 1+5, 2+6, 3+7
    __syncwarp();
    cache[tid] += cache[tid+2];  // 0+2, 1+3
    __syncwarp();
    cache[tid] += cache[tid+1];  // 0+1
    __syncwarp();    // 上述流程每次用前一半的线程
}

__global__ void reduce5(int *d_in,int *d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x]; // 从全局内存中多添加一个元素
    __syncthreads();

    // do reduction in shared mem, reduce的总数大于32时仍然采用之前的reduce方式
    for(unsigned int s=blockDim.x/2; s>=32; s>>=1){ 
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem 小于等于32时，就直接在warp内做reduce
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

// 求和reduce
int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp cudaprop;
    cudaGetDeviceProperties(&cudaprop, dev);
    cudaSetDevice(dev);

    int nElem = THREAD_PER_BLOCK*4; // 128*4
    int nSize = nElem * sizeof(int);
    int host_sum = 0;

    int *host = (int *)malloc(nSize);
    for(int i=0;i<nElem;i++){ // 0, 1, ..., n
        host[i] = i;
        host_sum+=i;
        // printf("%d ", host[i]);
    }
    // printf("\n");

    int *res = (int *)malloc(nSize);

    int *d_in, *d_out;
    cudaMalloc((int **)&d_in, nSize);
    cudaMalloc((int **)&d_out, nSize);

    dim3 block(THREAD_PER_BLOCK); // 128个线程
    dim3 grid ((nElem + block.x -1)/block.x); // 4个block

    double start = seconds();

    cudaMemcpy(d_in, host, nSize, cudaMemcpyHostToDevice);

    reduce4<<<grid, block>>>(d_in,d_out);

    cudaMemcpy(res, d_out, nSize, cudaMemcpyDeviceToHost);

    double eLpes = seconds() - start;
    printf("%f s\n", eLpes);

    int arraySum = 0;
    for(int i=0;i<4;i++){ // 每个block的计算结果
        printf("%d: %d \n", i, res[i]);
        arraySum += res[i];
    } 
    std::cout<<"Kernel: " <<arraySum<<std::endl;
    std::cout<<"Answer: "<< host_sum<<std::endl;

}