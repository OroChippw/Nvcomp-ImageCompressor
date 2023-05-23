#pragma once
#include <iostream>

struct CompressConfiguration {
    std::string input_dir;
    std::string output_dir;
    std::string rebuild_dir;

    int width = 8320;
    int height = 40000;

    std::string compresstype = "LZ4";
};