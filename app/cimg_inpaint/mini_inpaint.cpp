//
//  mini_inpaint.cpp
//  
//
//  Created by Carson Reid on 2020-03-20.
//

#include <stdio.h>
#define cimg_plugin "inpaint.h"
#include "CImg.h"
using namespace cimg_library;

int main(int argc, char **argv) {
    cimg_usage("Inpainting for ECE471");
    const char *file_i = cimg_option("-i", (char*)NULL, "Input Image");
    const char *mask_file = cimg_option("-m", (char*)NULL, "Mask Image");
    const char *file_o = cimg_option("-o", (char*)NULL, "Output File");
    if (file_i == NULL || mask_file == NULL || file_o == NULL) {
        std::printf("Invalid options supplied to inpaint.\n");
        std::fflush(stdout);
        return 1;
    }
    CImg<> im = CImg<>(file_i);
    CImg<int> mask = CImg<int>(mask_file);
    im.inpaint_patch(mask, 5);
    im.save(file_o);
    std::printf("Yay!\n");
    return 0;
}
