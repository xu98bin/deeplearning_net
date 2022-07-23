# deeplearning_net
用c++从头创建一个深度学习网络实现手写数字识别。如果有什么错误的地方希望大家能够帮忙改进。
程序是再msvc编译器下，使用vs2022写的，没有使用CMake。

更新2022/6/10：上传utils.hpp文件。utils.hpp文件包括二维矩阵的乘法、二维转置、矩阵点乘(不局限于二维)，对矩阵乘法进行了一定的优化。matrix_mm_slow函数是没有进行优化的矩阵乘法函数，matrix_mm是常规优化的函数，matrix_mm_fast使用avx2或者avx对乘法进行加速。以2048x2048 float类型的矩阵乘法为例子，matrix_mm_slow用时大概53s，matrix_mm用时大概4.2s,matrix_mm_fast用时1.4s,开启openMP后，matrix_mm_fast用时0.3s左右。matrix_dot是矩阵点成相关的函数，matrix_dot_fast使用了AVX，速度是没用使用AVX的2.5倍左右。

更新2022/7/23 第一个版本的代码初步实现完成。这是跑MNIST的结果。如果开启openmp，一个epoch训练时间大概90s。我测试了一下相同网络的pytorch手写数字识别，一个epoch大概30s。
![aaee7f53aa29c4e9573f43b5277a787](https://user-images.githubusercontent.com/78574951/180587558-d86c6249-f966-4219-a11a-d0289908e7ab.jpg)
