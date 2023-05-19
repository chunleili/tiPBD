import taichi as ti

ti.init(debug=True)
shape = (3, 3)
new_shape = (shape[0] + 2, shape[1] + 2)
val = ti.field(dtype=ti.f32, shape=(new_shape), offset=(-1, -1))
val.fill(1.0)
print(val)

# Output:
# [Taichi] version 1.6.0, llvm 15.0.1, commit 4c3e7cd3, win, python 3.10.9[Taichi] Starting on arch=x64
# Traceback (most recent call last):
#   File "d:\Dev\tiPBD\misc\code_bak\offset.py", line 7, in <module>
#     print(val)
#   File "C:\Users\GRBJ200045\.conda\envs\ti-build\lib\site-packages\taichi\lang\field.py", line 240, in __str__
#     return str(self.to_numpy())
#   File "C:\Users\GRBJ200045\.conda\envs\ti-build\lib\site-packages\taichi\lang\util.py", line 311, in wrapped
#     return func(*args, **kwargs)
#   File "C:\Users\GRBJ200045\.conda\envs\ti-build\lib\site-packages\taichi\lang\field.py", line 302, in to_numpy
#     tensor_to_ext_arr(self, arr)
#   File "C:\Users\GRBJ200045\.conda\envs\ti-build\lib\site-packages\taichi\lang\kernel_impl.py", line 1033, in wrapped
#     raise type(e)('\n' + str(e)) from None
# taichi.lang.exception.TaichiAssertionError:
# [kernel=tensor_to_ext_arr_c6_0] Out of bound access to ndarray at arg 0
# with indices [-1, -1]
# File "C:\Users\GRBJ200045\.conda\envs\ti-build\lib\site-packages\taichi\_kernels.py", line 42, in tensor_to_ext_arr:
#         arr[I] = tensor[I]
