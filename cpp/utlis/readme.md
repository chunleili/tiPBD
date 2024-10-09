calculate ridgid body mode using C++ and pybind11

# Download external libraries
```
git clone https://github.com/pybind/pybind11.git --depth=1
```

```
git clone https://gitlab.com/libeigen/eigen --depth=1
```

# Build pybind cpp code
(must has the same version of python as the one used to build pybind11)
Suppose we are in dir: tiPBD/cpp/rigidbodymode

```
conda activate py310
```

Remove-Item .\build\* -Recurse

```
cmake -B build
cmake --build build --config Release
```


## 计算染色，用于并行高斯赛德尔迭代法

输入：模型（.node格式）
输出：每个constraint的颜色

Reference:
[Parallel Block Neo-Hookean XPBD using Graph Clustering](https://profs.etsmtl.ca/sandrews/publication/xpbd_mig2022/)


## red-black GS
来源：
https://github.com/rpandya1990/Gauss-seidel-Parallel-Implementation

从vs 2022 native tool command prompt编译
```
cl.exe parallel_rb.cpp /openmp
```
结果
(Serial): 5975(ms)
(Parallel): 938(ms)