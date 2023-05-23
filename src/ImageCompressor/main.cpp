#include <iostream>
#include <opencv2/core.hpp>

#include "../ImageCompressorDll/CompressConfig.h"
#include "../ImageCompressorDll/ImageCompressor.h"

int main()
{
    std::string inputFilePath = "..//data//org";
    std::string CompressOutputFilePath = "..//data//compress_result";
    std::string ReconstructedFilePath = "..//data//reconstruct_result";


    CompressConfiguration cfg;
    cfg.input_dir = inputFilePath;
    cfg.output_dir = CompressOutputFilePath;
    cfg.rebuild_dir = ReconstructedFilePath;

    /* Compress Samples */
    NvcompCompressRunner* compressor = new NvcompCompressRunner();
    compressor->compress(cfg);

    return EXIT_SUCCESS;
}