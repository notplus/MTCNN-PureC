# MTCNN-PureC
MTCNN face detection with C and Arm-CMSIS-DSP

## Introduction
The project is a MTCNN face detection based on C. To implement MTCNN in MCU, the project don't rely on any library and you can use [ARM-CMSIS-DSP](https://github.com/ARM-software/CMSIS_5) to accelerate the matrix multiplication.
The input data is a RGB HWC txt, and output is the keypoints of face and bounding box of face.

本项目为MTCNN的C实现，无依赖库，主要为arm平台的mcu编写，在部署时可通过[ARM-CMSIS-DSP](https://github.com/ARM-software/CMSIS_5)库来加速矩阵乘法运算。
输入为RGB HWC格式的文本文件，输出为人脸包围盒和人脸关键点。

## Demo
![picture 1](https://github.com/notplus/MTCNN-PureC/blob/master/1.jpg)
### output  
* boundingBox3 is 160 72 450 362         //(x1, y1, x2, y2)
* keyPoint0 is 259.010193 188.661850  
* keyPoint1 is 350.958679 183.481461   
* keyPoint2 is 309.558563 237.440964   
* keyPoint3 is 267.242126 291.008728   
* keyPoint4 is 354.936646 283.069946   

### print with opencv
![picture 2](https://github.com/notplus/MTCNN-PureC/blob/master/result.jpg)


## Implement Details
The project mainly refers to [MTCNN-light](https://github.com/AlphaQi/MTCNN-light) and remove OpenCV and OpenBLAS dependencies.      
Convolution is implemented by im2col and matrix multiplication.  

本项目主要参考了[MTCNN-light](https://github.com/AlphaQi/MTCNN-light)，并且移除了OpenCV和OpenBLAS依赖。  
