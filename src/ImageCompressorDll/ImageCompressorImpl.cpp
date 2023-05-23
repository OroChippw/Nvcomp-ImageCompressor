#pragma warning (disable:4819)

#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "ImageCompressorImpl.h"
#include "CompressConfig.h"

int NvcompCompressRunnerImpl::ReadInput(const std::string input_path)
{   
    std::cout << "=> Start ReadInput and build file lists ..." << std::endl;
    struct stat s;
    int error_code = 1;
    if(stat(input_path.c_str() , &s) == 0)
    {
        if(s.st_mode & S_IFREG)
        {
            files_list.emplace_back(input_path.c_str());
        }else if(s.st_mode & S_IFDIR)
        {
            struct dirent* dir;
            DIR* dir_handle = opendir(input_path.c_str());
            if(dir_handle)
            {
                error_code = 0;
                while((dir = readdir(dir_handle)) != NULL)
                {
                    if(dir->d_type == DT_REG)
                    {
                        std::string filename = input_path + "\\" + dir->d_name;
                        files_list.emplace_back(filename);
                    }else if(dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if(sname != "." && sname != "..")
                        {
                            ReadInput(input_path + sname + "\\");
                        }
                    }
                }
                closedir(dir_handle);
            }else
            {
                std::cout << "Can not open input directory : " << input_path << std::endl;
                return EXIT_FAILURE;
            }
        }else
        {
            std::cout << "Cannot find input path " << input_path << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Build file lists successfully ..." << std::endl;
    std::cout << "Files list num : " << files_list.size() << std::endl;
    return EXIT_SUCCESS;
}

int NvcompCompressRunnerImpl::Compress(CompressConfiguration cfg)
{

}

double NvcompCompressRunnerImpl::CalculatePSNR(cv::Mat srcImage , cv::Mat compImage)
{
    const unsigned int w = srcImage.cols;
    const unsigned int h = srcImage.rows;
    const unsigned int max = 255;
    cv::Mat subImage;
    cv::absdiff(srcImage , compImage , subImage);
    subImage = subImage.mul(subImage);
    cv::Scalar sumScalar = sum(subImage);
    double sse = sumScalar.val[0] + sumScalar[1] + sumScalar[2];
    if(sse <= 1e-10)
    {
        return 0;
    }else
    {
        double mse = sse / h / w;
        double psnr = 10 * log10(pow(max , 2) / mse);
        std::cout << "[VAL->MSE] : " << mse << " [VAL->PSNR] : " << psnr << std::endl;
        return psnr;
    }
}

int NvcompCompressRunnerImpl::CompressImage(CompressConfiguration cfg)
{
    int read_state = ReadInput(cfg.input_dir);
    std::cout << "=> Start image compression ... " <<std::endl;
    if(Compress(cfg))
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}