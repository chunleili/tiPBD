# Download pybind11
```
git clone https://github.com/pybind/pybind11.git --depth=1
```

# Build pybind cpp code
(must has the same version of python as the one used to build pybind11)
Suppose we are in dir: tiPBD/pybind

```
conda activate py310
```

```
cmake -B build
cmake --build build --config Release
```