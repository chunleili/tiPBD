## 创建SDF
```python
# 2D SDF 5x5大小的
sdf = SDF((5, 5))
# 全部填充为1.0
sdf.val.fill(1)
# 打印sdf值
print(sdf.val)
# 打印sdf梯度
print(sdf.grad)
# 计算梯度
sdf.compute_gradient(1.0,1.0)
# 打印sdf
print(sdf)
# 将sdf写入文件
sdf.print_to_file()

# 3D SDF 5x5x5大小的
sdf_3d = SDF((5, 5, 5))
sdf_3d.val.fill(1)
print(sdf_3d.val)
print(sdf_3d.grad)
sdf_3d.compute_gradient(1.0,1.0,1.0)
print(sdf_3d)
sdf_3d.print_to_file("result/sdf_3d")
```


## 从网格生成SDF
