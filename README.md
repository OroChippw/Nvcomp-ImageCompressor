# Nvcomp-ImageCompressor
## Introduction
Nvcomp ImageCompressor is to create a C++ interface that enables direct invocation of nvcomp libraries.The nvcomp is a CUDA library that features generic compression interfaces to enable developers to use high-performance GPU compressors and decompressors in their applications. This repository is used to record the ablation experiment data that use nvcomp for image compression.The code repository contains a C++ library with a test program to facilitate easy integration of the interface into other projects.

Currently, the interface only supports GPU execution.The specific experimental data and equipment used are shown below. And the inferface is only supported on Windows and may encounter issues when running on Linux.

## Development Enviroments
>  - Windows 10 Professional 
>  - CUDA v11.3
>  - cmake version 3.26.2

## Quick Start

### Requirements
``` 
# CUDA 3rdparty
the cuda_runtime.h usually placed in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include . You can create a folder called cuda113 and copy include and lib into it
# opencv 3rdparty
This repository use opencv 4.x
# nvcomp 3rd
This repository use nvcomp_2.6.1_windows_12.x
# CXX_STANDARD 17
```

### How to build and run
```
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# Enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```

## Ablation Experiment Record
Environment Device : i5-13600KF + NVIDIA GeForce RTX 3060（12GB）
Input Image : [sample A] : 1.png 525MB ; [sample B] : 6.bmp 952MB ; [sample C] : 11.png 583MB
all image resolution is 8320*40000
### ② Different Compression Algorithms

| Compression Algorithms | meanCompressCostTime(ms) | meanDecompressCostTime(ms) | compression Ratio(%) |
| :--------------:| :------------: | :------------: | :----------------: |
| LZ4   | 749.02 | 56.24 | A : 1.0 ; B : 0.60 ; C : 1.0 |
| Snappy | 375.60 | 97.38 | A : 1.0 ; B : 0.60 ; C : 1.0 |
| BitComp | 17.36 | 15.66 | A : 1.0 ; B : 0.60 ; C : 1.0 |
| ANS | 23.15 | 15.91 | A : 1.0 ; B : 0.60 ; C : 1.0 |
| Cascade | 52.13 | 17.38 | A : 1.0 ; B : 0.60 ; C : 1.0 |
| Gdeflate | 890.06 | 60.78 | A : 1.0 ; B : 0.60 ; C : 1.0 |

Conclusion: The compression ratios achieved by different compression algorithms on this batch of data are the same, the main difference lies in the compression speed and decompression speed  

### ③ Generate 8432*40000 pure color bmp image to compress
| Compression Algorithms | meanCompressCostTime(ms) | meanDecompressCostTime(ms) | compression Ratio(%) |
| :--------------:| :------------: | :------------: | :----------------: |
| LZ4   | 32.55 | 7.84 | 964->1.02[0.105] |

### License
This project is licensed under the MIT License.