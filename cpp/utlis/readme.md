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
