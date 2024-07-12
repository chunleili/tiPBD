use cmd(not powershell) to compile
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

run cloth3d.py to get G.npz

run test_spmm.py: run_spgemm() to run cuda spmm
The sparse matrix C will output in C.data.txt, C.indices.txt, C.indptr.txt

run test_spmm.py: validify() to compare results