#pragma once
#pragma warning (disable:4819)

#include <iostream>
#include <opencv2/core.hpp>
#include <nvcomp.h>

#include "dirent.h"
#include "CompressConfig.h"

#define CHECK_CUDA(call)                                                    \
{                                                                           \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
    {                                                                       \
        std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                            \
    }                                                                       \
}

class NvcompCompressRunnerImpl
{
private:
    cv::Mat compress_image;
    std::vector<std::string> files_list;

private:
    /* 通过和原图对比计算均方误差和峰值信噪比以评估图像质量 */
    double CalculatePSNR(cv::Mat srcImage , cv::Mat compImage);

public:
    NvcompCompressRunnerImpl() {};
    ~NvcompCompressRunnerImpl() {};

public:
    int ReadInput(const std::string input_path);
    int Compress(CompressConfiguration cfg);

public:
    int CompressImage(CompressConfiguration cfg);

};