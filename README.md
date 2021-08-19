# MTCNN-PureC
MTCNN face detection with C and Arm-CMSIS-DSP

## Introduction
The project is a MTCNN face detection based on C. To implement MTCNN in MCU, the project doesn't rely on any library and you can use [ARM-CMSIS-DSP](https://github.com/ARM-software/CMSIS_5) to accelerate the matrix multiplication.
The input data is a RGB CHW txt, and output is the keypoints of face and bounding box of face.

本项目为MTCNN的C实现，无依赖库，主要为arm平台的mcu编写，在部署时可通过[ARM-CMSIS-DSP](https://github.com/ARM-software/CMSIS_5)库来加速矩阵乘法运算，请参考官方文档。

本项目在保持API不变下简化了`arm_math.h`和`arm_mat_mult_f32.c`，以便在PC平台运行。

输入为RGB CHW格式的文本文件，输出为人脸包围盒和人脸关键点。

## Usage
```shell
git clone https://github.com/notplus/MTCNN-PureC.git
cd MTCNN-PureC
mkdir build
cd build
cmake ..
make
```

## Demo
![picture 1](https://github.com/notplus/MTCNN-PureC/blob/master/1.jpg)

### Input
输入为RGB CHW 格式图像生成的文本文件，可使用如下代码生成，另外使用时需要在`include/test.h`中修改图像尺寸。

```python
import numpy as np
import cv2
img = cv2.imread("1.jpg")
img_ = img[:,:,::-1].transpose((2,0,1))
np.savetxt('input1.txt', img_.reshape(-1), "%d")
```

### Output  
* boundingBox3 is 160 72 450 362         //(x1, y1, x2, y2)
* keyPoint0 is 259.010193 188.661850  
* keyPoint1 is 350.958679 183.481461   
* keyPoint2 is 309.558563 237.440964   
* keyPoint3 is 267.242126 291.008728   
* keyPoint4 is 354.936646 283.069946   

### Print with opencv
![picture 2](https://github.com/notplus/MTCNN-PureC/blob/master/result.jpg)


## Implement Details
The project mainly refers to [MTCNN-light](https://github.com/AlphaQi/MTCNN-light) and remove OpenCV and OpenBLAS dependencies.      
Convolution is implemented by im2col and matrix multiplication.  

本项目主要参考了[MTCNN-light](https://github.com/AlphaQi/MTCNN-light)，并且移除了OpenCV和OpenBLAS依赖。  
