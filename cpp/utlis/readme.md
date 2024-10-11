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

## multi_color_gauss_seidel
来源：https://gist.github.com/Erkaman/b34b3531e209a1db38e259ea53ff0be9#file-gauss_seidel_graph_coloring-cpp-L101

从vs 2022 native tool command prompt编译
```
cl.exe multi_color_gauss_seidel.cpp
```

## 并行图染色
来源： https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/

编译
```
cmake -B build -DUSE_ECL=1
cmake --build build --config Release
```

<!-- 编译
```
nvcc -O3 -arch=native ECL-GC_12.cu -o ecl-gc
``` -->

测试数据来源（internet.egr）：https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html

运行测试
```
./ecl-gc.exe internet.egr
```

### 生成数据：转换matrix market文件到.egr文件



<!-- 从vs 2022 native tool command prompt编译
```
cl.exe ./mm2ecl.cpp
``` -->

生成matrix market文件(根目录运行)
```
cd ../..
python engine/soft/soft3d.py -export_matrix=1 -end_frame=1 -export_matrix_binary=0
cp result/latest/A_F0.mtx cpp/ulits/A.mtx
```


```
./mm2ecl.exe A.mtx
```