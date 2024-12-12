# Accelerating with CUDA
If you want to use the CUDA version of the fast multigrid, your should use `-use_cuda=1` in python script, and compile the CUDA code in `cpp/mgcg_cuda`. Python will load dlls in `cpp/mgcg_cuda/lib`.

# Compile CUDA code
1. Install CUDA. My version is 12.5, VS2022
2. Change set(CMAKE_CUDA_ARCHITECTURES 89) in CMakeLists.txt
3. Run cmake from **cmd**(Not powershell! Powershell does not work for `set(CMAKE_CUDA_ARCHITECTURES 89).`)
4. Compile fastmg.cu by cmake and generate dlls in `cpp\mgcg_cuda\lib`.
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Output in `lib/*`。

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin`

5. Change `os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")` as your own path in python code.


Note: In Windows, you can use `dumpbin /dependents` to check the dependencies of a dll. For example, `dumpbin /dependents fastmg.dll` will show the dependencies of fastmg.dll.


Download(600MB): 
https://bhpan.buaa.edu.cn/link/AA1B784714F4D846E59851D1B12E83196D
or
https://www.dropbox.com/scl/fi/tvpr3g3btjca0maaz6gfl/dlls.zip?rlkey=68u687oghe5i7tddvedggcy7l&st=eo8dg8cd&dl=0


# download external libraries
```
   git submodule update --init 
```


# misc
## Remove build files in powershell
```
Remove-Item .\build\* -Recurse
```

## other 
Run directly from "x64 Native Tools Command Prompt for VS 2022"

nvcc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5/include" -I"." fastmg.cu -o fastmg -lcusparse


## Debug in msvc

 https://learn.microsoft.com/zh-cn/visualstudio/python/debugging-mixed-mode-c-cpp-python-in-visual-studio?view=vs-2022 


## Debug in vscode
Install pythoncpp-debug plug-in: https://marketplace.visualstudio.com/items?itemName=benjamin-simmonds.pythoncpp-debug

Modify launch.json as in .vscode/launch.json


More Details:
(https://blog.csdn.net/weixin_43940314/article/details/141869353)


## python binding
in ./cpp
```
pip install mgcg_cuda
```


```python
import pymgpbd
```