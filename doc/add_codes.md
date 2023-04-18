## 自定义程序

为了遵循高内聚低耦合的原则，本程序将遵循以下约定：
1. 新增的功能不应该影响旧的功能
2. 程序运行的必须参数要尽量少。获取参数要尽量延后。
3. 对新增代码，可以复用原有代码，但也可以选择不用。不用的时候旧代码不应该对新代码有过多限制。


因此，所有的功能模块尽量以搭积木的方式使用。例如要实现一个新的软体模拟NewSoftBody，本程序提供的积木为：
1. GGUI可视化。需要更改solver_main.py。在solver_main函数开头的if语句中增加import new_softbody, 然后pbd_solver = NewSoftBody()。NewSoftBody实例至少要实现substep方法，并且至少要有pos_show属性，代表粒子的位置。如果要显示三角面，还可以定义indices_show属性，代表三角面的顶点索引。

2. 从json解析参数。json文件必须要有common和materials两个顶层的key。分别代表通用的参数和某个物体特有的参数。其中materials是个list，因为可能会有多个物体。在new_softbody需要使用到参数的时候，from engine.metadata import meta。 然后使用meta.get_common("key名")和meta.get_materials("key名")获取参数。json文件请放在data/scene目录下。