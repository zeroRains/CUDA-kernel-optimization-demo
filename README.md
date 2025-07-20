# CUDA Kernal基本优化思路DEMO

本仓库用于记录一些kernel的优化思路的一些实现demo，以便于个人理解和学习。

其中包括
1. softmax
2. reduce
3. sgemm的各类优化，2维线程块划分，向量化读取，warp级分块，双缓冲区
4. flash attention v1

有些kernel还配套创建了`XX_try.cu`文件，在理解了kernel实现细节之后可以在`XX_try.cu`文件中删除掉带有`core kernel` Tag的方法的实现内容，然后自己去实现以便，如果实现无误会显示结果正确的输出。

## 使用方法：
本仓库提供了基本的编译脚本`make_shell.sh`
```shell

# 使用方式
make_shell.sh [file_path]

# example
make_shell.sh falsh_attn/fa1.cu
# 会生成fa1.cu.o的文件
./flash_attn/fa1.cu.o
# Max shared memory: 49152, requested shared memory: 26624 
# Results are correct! 
```

本仓库参考了[cuda.keter.top](https://cuda.keter.top/)，感谢这位大佬提供的技术知识支持。