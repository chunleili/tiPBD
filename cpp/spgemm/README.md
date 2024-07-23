# cusparse spgemm called from python(ctypes)

1. Open cmd(not powershell) to compile using cmake
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
(If fails, CMAKE_CUDA_ARCHITECTURES might be changed for your own cuda architecture)

2. Change os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin") as your own cuda path

3. Run spgemm.py

Caution: If you run into access violation error. That probably because you input a small C nnz size. We use A.shape[0]*100 for now.


Reference:
- https://github.com/NVIDIA/CUDALibrarySamples/blob/ade391a17672d26e55429035450bc44afd277d34/cuSPARSE/spgemm/spgemm_example.c#L161
- https://docs.nvidia.com/cuda/cusparse/#cusparsespgemm