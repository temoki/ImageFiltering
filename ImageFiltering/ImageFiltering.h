//
//  ImageFiltering.h
//  ImageFiltering
//
//  Created by temoki on 2015/01/11.
//  Copyright (c) 2015年 temoki. All rights reserved.
//

#ifndef ImageFiltering_ImageFiltering_h
#define ImageFiltering_ImageFiltering_h

#include <stdint.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreVideo/CoreVideo.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ConvolutionFilter {
    const int16_t * kernel;
    int16_t         size;
    int16_t         divisor;
} ConvolutionFilter;

ConvolutionFilter GaussianFilter3x3();
ConvolutionFilter GaussianFilter5x5();
ConvolutionFilter GaussianFilter7x7();

CGImageRef CreateImage(CVImageBufferRef buffer);
CGImageRef CreateFilteredImage_CPU(CVImageBufferRef buffer, ConvolutionFilter filter, bool neon);
CGImageRef CreateFilteredImage_CPU_MultiThread(CVImageBufferRef buffer, ConvolutionFilter filter, bool neon, size_t numOfThreads);
CGImageRef CreateFilteredImage_GPU_vImage(CVImageBufferRef buffer, ConvolutionFilter filter);

#ifdef __cplusplus
}
#endif
    
#endif
