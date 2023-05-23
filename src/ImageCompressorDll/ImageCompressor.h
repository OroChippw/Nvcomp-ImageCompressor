#pragma once
#include <iostream>
#include "CompressConfig.h"

#ifdef NVCOMP_COMPRESS_RUNNER_EXPORTS
#define NVCOMP_COMPRESS_RUNNER_API __declspec(dllexport)
#else
#define NVCOMP_COMPRESS_RUNNER_API __declspec(dllimport)
#endif


#pragma warning(push)

class NvcompCompressRunnerImpl;

class NVCOMP_COMPRESS_RUNNER_API NvcompCompressRunner{
private:
    NvcompCompressRunnerImpl* compressor;

public:
    NvcompCompressRunner();
    ~NvcompCompressRunner();
    void compress(CompressConfiguration cfg);
};

#pragma warning(pop)