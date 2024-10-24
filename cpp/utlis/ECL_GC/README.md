打开终端，输入

```
nvcc -O3 ECL-GC-ColorReduction_12.cu -o ecl-gc-ColorReduction
```

进行编译

输入

```
./ecl-gc-ColorReduction ele 
```

运行

将模型名称改为input.ele，然后将color.txt复制回到data/model/bunny_small下

或运行
```
python .\cpp\utlis\ECL_GC\auto.py E:\Dev\tiPBD\data\model\bunny85w\bunny85w.ele
```