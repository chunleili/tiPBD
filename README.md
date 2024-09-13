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
- `-solver_type=xpbd` 使用xpbd求解器而非AMG
- `-cuda_dir` 指定cuda安装目录。默认为`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin`
- `-out_dir` 输出目录，默认为`result/latest`
- `-auto_another_outdir=1` 每次运行自动创建一个新的输出目录，防止覆盖之前的结果。例如`result/latest_1`
- `-export_residual=1` （默认为0）每帧输出残差到文件，例如`result/latest/r/1.json`。
- `-end_frame=1000` 结束帧数
- `-setup_interval=20` 每隔多少帧更新一次AMG setup phase
- `-max_iter=1000` 每帧最大迭代次数