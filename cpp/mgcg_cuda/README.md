# Accelerating with CUDA
If you want to use the CUDA version of the fast multigrid, your should use `-use_cuda=1` in python script, and compile the CUDA code in `cpp/mgcg_cuda`. Python will load dlls in `cpp/mgcg_cuda/lib`.

# Compile CUDA code

## Too long don't read

Run `builcuda.bat`

It will appear in `cpp\mgcg_cuda\lib`

When you run the python code, make sure 

1. Add libdir to PYTHONPATH by `sys.path.append(cpp\mgcg_cuda\lib)`.

2. Let the python find the cuda by:
   - (Windows)`os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")`.
   - (Linux) Add the cuda dir to system path (which usually has been done automatically when installing cuda).



## Detailed Steps
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

5. Change `os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")` as your own path in python code (only Windows needed).


Note: In Windows, you can use `dumpbin /dependents` to check the dependencies of a dll. For example, `dumpbin /dependents fastmg.dll` will show the dependencies of fastmg.dll.


<!-- Download(600MB): 
https://bhpan.buaa.edu.cn/link/AA1B784714F4D846E59851D1B12E83196D
or
https://www.dropbox.com/scl/fi/tvpr3g3btjca0maaz6gfl/dlls.zip?rlkey=68u687oghe5i7tddvedggcy7l&st=eo8dg8cd&dl=0 -->


# download external libraries(git submodule)
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


# python binding
Just run the cmake in mgcg_cuda folder.
```
cmake -B build
cmake --build build --config Release
```

Or run the `buildcuda.bat` from the root folder. It will compile and get something like `cpp\mgcg_cuda\lib\pymgpbd.cp310-win_amd64.pyd`, which is a dynamic libary for python.



## Fix the `ModuleNotFoundError: No module named 'pymgpbd'` error:

Before import module, add the lib dir to the PYTHONPATH by the following code:
```python
sys.path.append(libdir)
# where libdir is `<proj_dir>/cpp/mgcg_cuda/lib`

import pymgpbd as mp
```


## Fix VSCode python intellisense error

 Add the following to `./vscode/settings.json`
```json
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/cpp/mgcg_cuda/lib/"
    ],
    "python.analysis.extraPaths": [

        "${workspaceFolder}/cpp/mgcg_cuda/lib/"
    ],
```

## Install the module(optional)
In ./cpp

```
pip install ./mgcg_cuda
```

This is rely on the pyproject.toml file in the cpp folder.

It will install the module in the site-packages folder of the python environment.

If you want to uninstall the module, use `pip uninstall mgcg_cuda`.

Installing is not necessary. It may take long time. I suggest you to install it only if you have finished the development and want to publish the module.