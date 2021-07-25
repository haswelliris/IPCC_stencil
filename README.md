# IPCC_stencil
IPCC2020初赛题目：stencil
## 赛题内容： 9-point stencil 图像处理并行优化
详见 [2020 年首届 ACM 中国-国际并行计算挑战赛 全国初赛赛题.pdf](./2020年首届ACM中国-国际并行计算挑战赛全国初赛赛题.pdf)
## 编译&运行
需要gcc8以上  
注意不同Linux系统(rhel和debian)有关PIE的区别，参考`Makefile.gcc-debian`
### 编译
```
make
```
### 运行
```
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
chmod +x ./setup-omp.sh
./setup-omp.sh
./bin/stencil IPCC.png
```
### 提交运行
```
sbatch run.slurm
```
## 优化结果
### 运行设备
AMD EPYC 7452(32Cores @2.35Ghz) * 2 sockets  
DDR4 2933Mhz 16GB * 8 Channels / Socket
### 运行时间
|  | 10次总时间 | 
| :---- | ----: |
| 基准 | 10989.4ms |
| 优化后 | 15.8ms |
实现了**695.53**倍的加速
### 运行记录
```
build...
g++ -Iinclude/ -fopenmp -Ofast -mavx2 -mfma -g -o bin/stencil src/main.o src/image.o src/stencil.o lib/libpng16.a lib/libz.a

running...

[1mEdge detection with a 3x3 stencil[0m

Image size: 10410 x 5905

[1mTrial        Time, ms 
    1           2.385 
    2           1.526 
    3           1.491 
    4           1.480 
    5           1.495 
    6           1.485 
    7           1.493 
    8           1.484 
    9           1.486 
   10           1.481 
-----------------------------------------------------
Total :       15.8 ms
-----------------------------------------------------

Output written into data.txt and output.png

```
## 优化报告
详见 [stencil优化报告](./stencil优化报告.pdf)
