# taichi实现PBD软体模拟
这是一个用taichi实现的软体模拟小玩具。目的是为了测试PBD算法和理解PBD算法原理。因此代码非常简单。

# 参考
主要参考自Matthias Muller的tenMinutePhysics
https://matthias-research.github.io/pages/tenMinutePhysics/
尤其是其第9讲和第10讲。这里也参考了他的代码实现。他的代码是使用javascript + THREE.js写的。我们将其改为taichi实现。

# 运行方法
安装taichi>=1.04
运行
```
python tiPBD.py
```

将展示结果为：
![demo](demo.gif)
也就是一个下落的斯坦福兔子

# 文件结构
总共3个文件：
- tiPBD.py是主要逻辑。PBD算法在此文件。代码很短，只有130行左右。

- mesh_data.py保存的是斯坦福兔子的模型。按照OBJ的格式保存为一个python字典。其中包括四个字段：顶点位置、四面体顶点连接关系、边顶点连接关系和三角表面顶点连接关系。

- tiReadMesh.py是用来读入网格到taichi field当中的。主要使用pos保存顶点坐标，tet保存四面体顶点编号，edge保存边顶点编号，surf保存三角面顶点编号

关于tiReadMesh的详细用法，请参看
https://github.com/chunleili/tiReadMesh

# 原理
PBD是Matthias Muller等人发明的基于位置的动力学。它的算法流程就三步：
1. 根据外力和碰撞更新粒子，不考虑粒子间关系
2. 根据粒子间关系的约束，修正粒子的位置
3. 根据位置反向更新速度

实际上，本代码实现的是XPBD。它和PBD的区别很小，求解速度差不多，但是稳定性更好，并且支持更多类型的约束。

具体的可以参考这个博客
https://blog.csdn.net/weixin_43940314/article/details/126065813

还应该参考Muller等人的原始论文。
- PBD: https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
- XPBD: https://dl.acm.org/doi/10.1145/2994258.2994272