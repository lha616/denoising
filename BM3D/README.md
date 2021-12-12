# BM3D_python
BM3D的一个python实现

BM3D项目地址Matlab (http://www.cs.tut.fi/~foi/GCF-BM3D/)   version.

### 结果比较  (sigma = 15)


| verson, psnr | barbara.png 24.59  | boat.png 24.60 |  Cman.png 24.63 | couple.png 24.61 | fprint.png 24.59 | house.png 24.60 | Lena512.png  24.61 | man.png 24.61 | montage.png 24.61 | peppers.png 24.58 |
| :-: | :-: | :-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |
| BM3D(this) | 32.97 | 32.10 | 31.80 | 32.08 | 30.33 | 34.87 | 34.23 | 31.93 | 34.88 | 32.61 |
| BM3D(matlab) | 33.11 | 32.14 | 31.91 | 32.11|30.28|34.94|34.27|31.93|35.15 | 32.70|

第二行数据源于(http://www.cs.tut.fi/~foi/GCF-BM3D/) 

data ： 图像集  
result ：初步估计 和 最终估计

环境： python3.8 

运行：python bm3d.py