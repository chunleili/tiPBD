首先编译AMGX, 在AMGX目录下
```
cmake -B build
cmake --build build --config Release
```

编译完之后进入AMGX/build 运行demo. 注意这里比README多加了个Release
```
examples/Release/amgx_capi -m ../examples/matrix.mtx -c ../src/configs/FGMRES_AGGREGATION.json
```

然后设置
```
setx AMGX_DIR "D:\dev\AMGX"
```

然后在pyamgx下

```
pip install .
```

将pyamgx/demo.py的前几部分改为下面的代码，并且运行`python demo.py`

```
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import os

os.add_dll_directory("D:/Dev/AMGX/build/Release")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
import pyamgx

pyamgx.initialize()

cfg = pyamgx.Config().create_from_file("D:/dev/AMGX/src/configs/AGGREGATION_JACOBI.json")
```

注意几点：
1. add_dll_directory是让windows找到DLL的。因为windows DLL不会去PATH里面搜索，所以要这样设置。
2. create_from_dict在windows也是无法使用的。会出现如下报错
3. 如果`pip install .`这一步出现了问题，那么可以将AMGX/build/Release下面的几个dll拷贝到AMGX/build/试一试
```
Error parsing config file: Error: Cannot read config file: C:\Users\cl-w\AppData\Local\Temp\tmpudy9b9x1
Traceback (most recent call last):
  File "D:\dev\pyamgx\demo.py", line 16, in <module>
    cfg = pyamgx.Config().create_from_dict({
  File "pyamgx\\Config.pyx", line 63, in pyamgx.Config.create_from_dict
    with tempfile.NamedTemporaryFile(mode='r+') as fp:
  File "pyamgx\\Config.pyx", line 66, in pyamgx.Config.create_from_dict
    self.create_from_file(fp.name.encode())
  File "pyamgx\\Config.pyx", line 49, in pyamgx.Config.create_from_file
    check_error(AMGX_config_create_from_file(&self.cfg, param_file))
  File "pyamgx\\Errors.pyx", line 62, in pyamgx.check_error
    raise AMGXError(get_error_string(err_code))
pyamgx.AMGXError: I/O error.
```