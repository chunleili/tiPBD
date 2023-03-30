# 简介

这是taichi pbd库。

需要taichi>= 1.4.1和meshtaichi

使用方法：

在data/scene中添加场景文件，在model中添加几何文件（现在仅支持tetgen格式），然后从根目录运行`main.py`。

```python
python main.py
```

![select](doc/img/select.png)

![demo](doc/img/demo.png)


## 文件结构

- main # 主程序入口
- data
  - scene # 场景文件
  - model # 模型文件
- engine # 引擎
  - solver # 求解器, 用于实例化具体的pbd solver，以及ggui
  - metadata # 元数据: 几乎是所有数据的中转中心，你可以从这里获取一切参数，包括场景、模型、网格、材料参数等。
  - fem # 基于有限元的PBD。
    - fem_base # 有限元基类 除了约束投影以外的所有步骤
    - arap: # as rigid as possible 本构模型的约束投影
    - neohooken: # neo-hooken模型的约束投影
    - mesh: # 基于meshtaichi的网格数据
- ui # 用户界面
  - parse_commandline_args: # 解析命令行参数
  - config_builder # 用于从json场景文件构建配置文件，传给metadata
  - filedialog # 打开文件对话框以选择json场景文件

