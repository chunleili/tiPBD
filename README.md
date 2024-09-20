# Multi-grid PBD

## 运行
始终处于项目根目录

布料仿真
```
python engine/cloth/cloth3d.py
```

软体仿真
```
python engine/soft/soft3d.py
```

## 编译cuda
cd到cpp/mgcg_cuda
```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
详情见cpp/mgcg_cuda/README.md

运行python脚本前需要用`-cuda_dir=xxx` 指定cuda安装目录。

`-use_cuda=0`可以只运行python版本，不需要编译。建议先用python版本进行较小规模测试，例如N=64。


## 命令行选项
用法例如
```
python engine/cloth/cloth3d.py -N=64
```

- `-N=1024` 设定布料分辨率。顶点数为N^2。
- `-use_cuda=1` 使用cuda，默认使用。
- `-solver_type=XPBD` 使用xpbd求解器而非AMG
- `-cuda_dir` 指定cuda安装目录。默认为`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin`
- `-out_dir` 输出目录，默认为`result/latest`
- `-auto_another_outdir=1` 每次运行自动创建一个新的输出目录，防止覆盖之前的结果。例如`result/latest_1`
- `-export_residual=1` （默认为0）每帧输出残差到文件，例如`result/latest/r/1.json`。
- `-end_frame=1000` 结束帧数
- `-setup_interval=20` 每隔多少帧更新一次AMG setup phase
- `-maxiter=1000` 每帧最大迭代次数
- `-maxiter_Axb=100` 每个Ax=b最大迭代次数


## 模型分辨率
建议在初次运行时采用较小的分辨率尝试。
- 布料：-N=64为较小规模，N=1024为较大规模。
- 软体：-model_path=xxx 指定模型文件路径。
    - 最小 8 vertices "data/model/cube/minicube.node"
    - 其次 1353 vertices "data/model/bunny1k2k/coarse.node"
    - 较大 27w vertices "data/model/bunnyBig/bunnyBig.node"
    - 最大 85w vertices"data/model/bunny85w/bunny85w.node"


### 输出结果
- `result/latest` 为默认的最新的输出目录。
  - `result/latest/mesh/1.ply` 输出每帧的网格。可用windows自带3D查看器查看，或者加载到Houdini查看。
  - `result/latest/latest.log` 打印到终端的输出同时也会输出到该文件。可用于后处理。
  - `result/latest/r/1.json` 如果`-export_residual=1`，每帧残差输出。
  - `result/latest/A/` 如果`-export_matrix=1`，每帧输出A矩阵和b到该目录。用于观察矩阵。可用`A = scipy.sparse.load_npz("A.npz") `和 `b = np.load("b.npy")` 读取。
  - `result/latest/state/` 如果`-export_state=1`，每帧输出状态（包括顶点位置等）到该目录。用于重启。`-restart=1`可以开启重启。配合`-restart_frame=100`指定重启帧数。`restart_dir`指定重启目录。


## 调试（VSCode）
**调试时输入命令行选项**
.vscode/launch.json中的"Python: Current File" args选项中添加。

**调试python+cpp**
1）安装插件python c++ debugger。2）以debug模式编译cpp。3）左侧调试选python C++ Debug之后F5启动。

## profiling
 **cProfile**

使用cProfile进行性能分析python代码。
```python -m cProfile -o profile xxx.py```
输出profile文件后，用snakeviz查看
```snakeviz profile```

（仅可分析python函数的耗时，对于cpp代码无效）

 **time.perf_counter()**
 在python中打印出
 ```python
    tic = time.perf_counter()
    xxxx
    logging.info(f"    xxx time:{time.perf_counter()-tic}")
```

**GpuTimer**

见fastmg.cu中的GpuTimer类。用于测量cuda代码的运行时间。