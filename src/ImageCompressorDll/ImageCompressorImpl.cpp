#pragma warning (disable:4819)

#include <iostream>
#include<nvcomp.hpp>
#include<random>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include<nvcomp/lz4.hpp>
#include<nvcomp/nvcompManagerFactory.hpp>

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
     std::cout << "# ----------------------------------------------- #" << std::endl;
    for(unsigned int index = 0 ; index < files_list.size() ; index++)
    {
        std::cout << "=> Processing: " << files_list[index] << std::endl;
        std::string::size_type iPos = files_list[index].find_last_of('\\') + 1;
        std::string filename = files_list[index].substr(iPos, files_list[index].length() - iPos);
        std::string name = filename.substr(0, filename.find("."));
        std::string image_path = files_list[index];

        cv::Mat srcImage = cv::imread(image_path , cv::IMREAD_COLOR);

        size_t input_element_count = srcImage.rows * srcImage.cols * srcImage.channels();

        // 初始化事件及流
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        cudaEvent_t ev_start = NULL, ev_end = NULL;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_end));

        // 设置一个块大小为2^16==65536，块大小越小，吞吐量越高
        const int chunk_size = 1 << 16;
        nvcompType_t data_type = NVCOMP_TYPE_CHAR; // 1B

        // 获取剩余显存及总显存信息
        size_t freeMem, totalMem;
        CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
        if (freeMem < sizeof(uint8_t) * input_element_count) {
            std::cout << "=> Insufficient GPU memory to perform compression." << std::endl;
            exit(1);
        }

        const size_t in_bytes = sizeof(uint8_t) * input_element_count;
        std::cout << "Before compress : " << in_bytes << "B" << std::endl;
        // 将输入的数据放到显存中
        uint8_t* d_in_data;
        CHECK_CUDA(cudaMalloc(&d_in_data, in_bytes));
        CHECK_CUDA(cudaMemcpy(d_in_data, srcImage.data, in_bytes, cudaMemcpyHostToDevice));

        int gpu_num = 0;

        std::cout << "=> Decomp_compressed_with_LZ4Manager" << std::endl;
	    nvcomp::LZ4Manager nvcomp_manager{ chunk_size , data_type , stream ,gpu_num , ComputeAndVerify};

        std::cout << "=> Decomp_compressed_with_SnappyManager" << std::endl;
        nvcomp::SnappyManager nvcomp_manager{ chunk_size , stream ,gpu_num , ComputeAndVerify };

        nvcomp::CompressionConfig comp_config = nvcomp_manager.configure_compression(in_bytes);

        size_t comp_out_bytes = comp_config.max_compressed_buffer_size;
        std::cout << "comp_config.max_compressed_buffer_size : " << comp_config.max_compressed_buffer_size << std::endl;
        if (comp_out_bytes < 0) {
            std::cout << "Output size must be greater than 0" << std::endl;
        }

        // 分配压缩输出缓冲区
        uint8_t* d_comp_out;
        CHECK_CUDA(cudaMalloc(&d_comp_out , comp_out_bytes));
        std::cout << "Compression memory (input+output+scratch) : " << (in_bytes + comp_out_bytes) << "B" << std::endl;

        // 进行压缩
        CHECK_CUDA(cudaEventRecord(ev_start , stream));
        nvcomp_manager.compress(d_in_data, d_comp_out, comp_config);
        CHECK_CUDA(cudaEventRecord(ev_end , stream));

        comp_out_bytes = nvcomp_manager.get_compressed_output_size(d_comp_out);

        float compress_cost_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&compress_cost_time_ms, ev_start, ev_end));
        std::cout << "=> Compress Cost time : " << compress_cost_time_ms << "ms" << std::endl;
        std::cout << "comp_size : " << comp_out_bytes << " , compressed ratio : " << std::fixed << std::setprecision(2)
            << (double)input_element_count * sizeof(uint8_t) / comp_out_bytes << std::endl;

        // 将输出流从GPU上移到cpu上
        uint8_t* d_comp_out_host = new uint8_t[comp_out_bytes];
        //CHECK_CUDA(cudaMalloc(&d_comp_out_host, comp_out_bytes));
        CHECK_CUDA(cudaMemcpy(d_comp_out_host, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToHost));
        std::cout << "=> CudaMemcpyDeviceToHost finish" << std::endl;

        // 解压缩
        nvcomp::DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(d_comp_out);
        // 分配输出缓冲区
        const size_t decomp_bytes = decomp_config.decomp_data_size;
        uint8_t* decomp_out_ptr;
        CHECK_CUDA(cudaMalloc(&decomp_out_ptr , decomp_bytes));
        std::cout << "decompression memory (input+output+temp) : "
            << (decomp_bytes + comp_out_bytes) << "B" << std::endl;

        CHECK_CUDA(cudaEventRecord(ev_start, stream));
        nvcomp_manager.decompress(decomp_out_ptr, d_comp_out, decomp_config);
        CHECK_CUDA(cudaEventRecord(ev_end, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float decompress_cost_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&decompress_cost_time_ms, ev_start, ev_end));
        std::cout << "=> Decompress Cost time : " << decompress_cost_time_ms << "ms" << std::endl;

        nvcompStatus_t final_status = *decomp_config.get_status();
        if (final_status == nvcompErrorBadChecksum) {
            std::cout << "One or more checksums were incorrect." << std::endl;
        }

        uint8_t* d_decomp_out_host = new uint8_t[decomp_bytes];
        CHECK_CUDA(cudaMemcpy(d_decomp_out_host, decomp_out_ptr, decomp_bytes, cudaMemcpyDeviceToHost));
        std::cout << "=> CudaMemcpyDeviceToHost finish" << std::endl;

        cv::Mat result_img(srcImage.rows, srcImage.cols, CV_8UC3, d_decomp_out_host);

        std::string::size_type iPos = cfg.input_dir.find_last_of('\\') + 1;
        std::string filename = cfg.input_dir.substr(iPos, cfg.input_dir.length() - iPos);
        //std::cout << filename << std::endl;
        std::string name = filename.substr(0, filename.find("."));

        std::string outputfile_path = cfg.output_dir + "\\LZ4\\" + name + ".png";
        cv::imwrite(outputfile_path, result_img);
        std::cout << "Save as : " << outputfile_path << std::endl;

        CHECK_CUDA(cudaStreamSynchronize(stream));

        // 用完进行销毁，避免显存泄露
        CHECK_CUDA(cudaFree(d_in_data));
        //CHECK_CUDA(cudaFree(d_comp_scratch));
        CHECK_CUDA(cudaFree(d_comp_out));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_end));
        CHECK_CUDA(cudaStreamDestroy(stream));

        delete[] d_decomp_out_host;
        delete[] d_comp_out_host;

        mean_compress_time += compress_cost_time_ms;
        mean_decompress_time += decompress_cost_time_ms;
    }
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