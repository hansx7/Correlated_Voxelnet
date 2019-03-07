import ctypes
from ctypes import *
import numpy as np

IoU = ctypes.cdll.LoadLibrary('./libIoU.so')
# IoU.IoU_interface1.argtypes = [c_double * 4, c_double * 4, c_double * 4, c_double * 4]
IoU.IoU_interface1.restype = c_double
# IoU.IoU_interface2.argtypes = [c_double * 5, c_double * 5]
IoU.IoU_interface2.restype = c_double


status_code = IoU.print_OK()
print(status_code, '\n')

rnd_data = np.array([[77.93, 69.15, 27.77, 08.86, 053.83],\
                     [14.21, 66.49, 04.92, 53.86, 143.35],\
                     [77.63, 00.59, 00.01, 00.01, 123.62],\
                     [57.36, 91.72, 34.26, 05.40, 059.26],\
                     [57.82, 64.29, 25.67, 53.68, 132.11],\
                     [31.35, 40.67, 51.23, 28.62, 095.30],\
                     [30.69, 30.58, 40.22, 98.02, 039.29],\
                     [80.42, 50.11, 84.56, 13.93, 161.67],\
                     [37.84, 49.19, 44.21, 73.73, 022.29],\
                     [43.70, 83.15, 43.24, 51.98, 165.37]])


for i in range(10):

	### pass in a single pair of rectangles

	INPUT = c_float * 5
	para1 = INPUT()
	for j in range(5):
		para1[j] = rnd_data[i][j]
	para2 = INPUT()
	for j in range(5):
		para2[j] = rnd_data[i][j]

	para1 = np.asarray(rnd_data[i])
	para1 = (ctypes.c_double * 5)(*para1)
	para2 = np.asarray(rnd_data[i])
	para2 = (ctypes.c_double * 5)(*para2)
	answer = IoU.IoU_interface2(para1, para2)
	print(answer)
	
	#######################################

### pass in two groups of rectangles

# para3 = rnd_data.tolist()

class Matrix(Structure):
	_fields_ = [('size_x', c_int), ('size_y', c_int), ('arr', c_double * 500000)]

IoU.IoU_interface3.restype = Matrix

para3 = []
for i in range(10):
	para3.append((ctypes.c_double * 5)(*rnd_data[i]))
para3 = ((ctypes.c_double * 5) * 10)(*para3)
answer = IoU.IoU_interface3(para3, para3, 10, 10)
print(answer.size_x, answer.size_y)
for i in range(answer.size_x):
	for j in range(answer.size_y):
		print(answer.arr[i*answer.size_y+j], )
	print()

####################################


