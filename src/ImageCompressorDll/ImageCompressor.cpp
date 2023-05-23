#include <iostream>

#include "ImageCompressor.h"
#include "ImageCompressorImpl.h"

NvcompCompressRunner::NvcompCompressRunner()
{
    compressor = new NvcompCompressRunnerImpl();
    std::cout << "=> Build NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

NvcompCompressRunner::~NvcompCompressRunner()
{
    delete compressor;
    std::cout << "=> Delete NvjpegCompressRunnerImpl successfully ..." << std::endl;
}

void NvcompCompressRunner::compress(CompressConfiguration cfg)
{
    std::string run_state = compressor->CompressImage(cfg) ? "Failure" : "Finish";
    std::cout << "=> Compress " << run_state << std::endl;
}
