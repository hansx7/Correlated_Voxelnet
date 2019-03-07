#IoU文件夹说明
###实现的文件
- IoU.h：IoU.c的头文件；
- IoU.c：实现计算两个矩形相交部分面积的c文件；
- ctypes.py：利用ctypes调用IoU.c文件的python文件；
- generate_rectangles.cpp：随机生成(x, y, w, h, r)的矩形；

###使用方法
利用如下命令（filename为IoU，libname为libIoU）编译c文件生成so文件：
```bash
gcc filename.c -fPIC -shared -o libname.so
```
然后运行ctypes_test.py（内含各种接口调用方法）即可。
