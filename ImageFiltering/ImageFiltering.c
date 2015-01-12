//
//  ImageFiltering.c
//  ImageFiltering
//
//  Created by temoki on 2015/01/11.
//  Copyright (c) 2015年 temoki. All rights reserved.
//

#include "ImageFiltering.h"
#include <stdlib.h>                 // malloc/free
#include <string.h>                 // memcpy
#include <limits.h>                 // UCHAR_MAX
#include <dispatch/dispatch.h>      // GCD
#include <Accelerate/Accelerate.h>  // vImage
#ifdef __ARM_NEON__
#include <arm_neon.h>               // ARM NEON
#endif

#ifndef MIN
#define MIN(a,b)	 (a < b ? a : b)
#endif
#ifndef MAX
#define MAX(a,b)	 (a < b ? a : b)
#endif

ConvolutionFilter GaussianFilter3x3() {
    static const int16_t KERNEL[] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    ConvolutionFilter filter = {KERNEL, 3, 16};
    return filter;
}

ConvolutionFilter GaussianFilter5x5() {
    static const int16_t KERNEL[] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    ConvolutionFilter filter = {KERNEL, 5, 256};
    return filter;
}

ConvolutionFilter GaussianFilter7x7() {
    static const int16_t KERNEL[] = {
         1,   6,  15,  20,  15,   6,  1,
         6,  36,  90, 120,  90,  36,  6,
        15,  90, 225, 300, 225,  90, 15,
        20, 120, 300, 400, 300, 120, 20,
        15,  90, 225, 300, 225,  90, 15,
         6,  36,  90, 120,  90,  36,  6,
         1,   6,  15,  20,  15,   6,  1
    };
    ConvolutionFilter filter = {KERNEL, 7, 4096};
    return filter;
}

CGImageRef CreateImage(CVImageBufferRef buffer) {
    const OSType type = CVPixelBufferGetPixelFormatType(buffer);
    const size_t width  = CVPixelBufferGetWidth(buffer);
    const size_t height = CVPixelBufferGetHeight(buffer);
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
    if (type != kCVPixelFormatType_32BGRA) return NULL;
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef cgContext = CGBitmapContextCreate(CVPixelBufferGetBaseAddress(buffer),
                                                   width, height, 8, bytesPerRow, colorSpace, bitmapInfo);
    CGImageRef outputImage = CGBitmapContextCreateImage(cgContext);
    
    CGContextRelease(cgContext);
    CGColorSpaceRelease(colorSpace);
    
    return outputImage;
}

CGImageRef CreateFilteredImage_CPU(CVImageBufferRef buffer, ConvolutionFilter filter, bool neon) {
    const OSType type = CVPixelBufferGetPixelFormatType(buffer);
    const size_t width  = CVPixelBufferGetWidth(buffer);
    const size_t height = CVPixelBufferGetHeight(buffer);
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
    const size_t bytesPerPixel = 4;
    const size_t bufferSize = CVPixelBufferGetDataSize(buffer);
    const size_t offset = filter.size / 2;
    if (type != kCVPixelFormatType_32BGRA) return NULL;
    if (bytesPerPixel != bytesPerRow / width) return NULL;
    
#ifndef __ARM_NEON__
    neon = false;
#endif
    
    const uint8_t * const inputBuffer = (uint8_t *)CVPixelBufferGetBaseAddress(buffer);
    uint8_t * outputBuffer = (uint8_t *)malloc(bufferSize);
    memcpy(outputBuffer, inputBuffer, bufferSize);
    
    for (size_t y = offset; y < height - offset; y++) {
        size_t y_head = y * bytesPerRow;
        for (size_t x = offset; x < width - offset; x++) {
            int32_t value[bytesPerPixel] = {0, 0, 0, 0};
            int32x4_t value4 = {0, 0, 0, 0};
            for (size_t ky = 0; ky < filter.size; ky++) {
                size_t iy = y - offset + ky;
                size_t iy_head = iy * bytesPerRow;
                size_t ky_head = ky * filter.size;
                for (size_t kx = 0; kx < filter.size; kx++) {
                    size_t ix = x - offset + kx;
                    size_t inputIndex = iy_head + (ix * bytesPerPixel);
                    size_t kernelIndex = ky_head + kx;
                    if (neon) {
                        int32x4_t kernel4 = vdupq_n_u32((int32_t)filter.kernel[kernelIndex]);
                        int32x4_t input4 = {inputBuffer[inputIndex], inputBuffer[inputIndex+1], inputBuffer[inputIndex+2], 0};
                        value4 = vaddq_s32(value4, vmulq_s32(input4, kernel4));
                    } else {
                        for (size_t c = 0; c < bytesPerPixel - 1; c++) {
                            value[c] += inputBuffer[inputIndex + c] * filter.kernel[kernelIndex];
                        }
                    }
                }
            }
            size_t outputIndex = y_head + (x * bytesPerPixel);
            if (neon) {
#ifdef __ARM_NEON__
                float32x4_t value4f = {
                    (float32_t)vgetq_lane_s32(value4, 0),
                    (float32_t)vgetq_lane_s32(value4, 1),
                    (float32_t)vgetq_lane_s32(value4, 2),
                    (float32_t)vgetq_lane_s32(value4, 3)
                };
                value4f = vabsq_f32(vdivq_f32(value4f, vdupq_n_f32((float32_t)filter.divisor)));
                value4f = vminq_f32(value4f, vdupq_n_f32((float32_t)UCHAR_MAX));
                outputBuffer[outputIndex + 0] = (uint8_t)vgetq_lane_f32(value4f, 0);
                outputBuffer[outputIndex + 1] = (uint8_t)vgetq_lane_f32(value4f, 1);
                outputBuffer[outputIndex + 2] = (uint8_t)vgetq_lane_f32(value4f, 2);
#endif
            } else {
                for (size_t c = 0; c < bytesPerPixel - 1; c++) {
                    value[c] /= filter.divisor;
                    value[c] = value[c] < UCHAR_MAX ? value[c] : UCHAR_MAX;
                    value[c] = value[c] < 0 ? 0 : value[c];
                    outputBuffer[outputIndex + c] = (uint8_t)value[c];
                }
            }
        }
    }

    CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef cgContext = CGBitmapContextCreate(outputBuffer, width, height, 8,
                                                   bytesPerRow, colorSpace, bitmapInfo);
    CGImageRef outputImage = CGBitmapContextCreateImage(cgContext);
    free(outputBuffer);
    
    CGContextRelease(cgContext);
    CGColorSpaceRelease(colorSpace);
    
    return outputImage;
}

CGImageRef CreateFilteredImage_CPU_MultiThread(CVImageBufferRef buffer, ConvolutionFilter filter, bool neon, size_t numOfThreads) {
    const OSType type = CVPixelBufferGetPixelFormatType(buffer);
    const size_t width  = CVPixelBufferGetWidth(buffer);
    const size_t height = CVPixelBufferGetHeight(buffer);
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
    const size_t bytesPerPixel = 4;
    const size_t bufferSize = CVPixelBufferGetDataSize(buffer);
    const size_t offset = filter.size / 2;
    if (type != kCVPixelFormatType_32BGRA) return NULL;
    if (bytesPerPixel != bytesPerRow / width) return NULL;
    
#ifndef __ARM_NEON__
    neon = false;
#endif
    const uint8_t * const inputBuffer = (uint8_t *)CVPixelBufferGetBaseAddress(buffer);
    uint8_t * outputBuffer = (uint8_t *)malloc(bufferSize);
    memcpy(outputBuffer, inputBuffer, bufferSize);
    
    size_t * begin = (size_t *)malloc(sizeof(size_t) * numOfThreads);
    size_t * end = (size_t *)malloc(sizeof(size_t) * numOfThreads);
    for (size_t index = 0; index < numOfThreads; index++) {
        begin[index] = (index == 0)? offset : end[index - 1] + 1;
        end[index] = (index == numOfThreads - 1) ? width - offset : (width / numOfThreads) * (index + 1);
    }
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    dispatch_apply(numOfThreads, queue, ^(size_t index) {
        for (size_t y = offset; y < height - offset; y++) {
            size_t y_head = y * bytesPerRow;
            for (size_t x = begin[index]; x < end[index]; x++) {
                int32_t value[bytesPerPixel] = {0, 0, 0, 0};
                int32x4_t value4 = {0, 0, 0, 0};
                for (size_t ky = 0; ky < filter.size; ky++) {
                    size_t iy = y - offset + ky;
                    size_t iy_head = iy * bytesPerRow;
                    size_t ky_head = ky * filter.size;
                    for (size_t kx = 0; kx < filter.size; kx++) {
                        size_t ix = x - offset + kx;
                        size_t inputIndex = iy_head + (ix * bytesPerPixel);
                        size_t kernelIndex = ky_head + kx;
                        if (neon) {
                            int32x4_t kernel4 = vdupq_n_u32((int32_t)filter.kernel[kernelIndex]);
                            int32x4_t input4 = {inputBuffer[inputIndex], inputBuffer[inputIndex+1], inputBuffer[inputIndex+2], 0};
                            value4 = vaddq_s32(value4, vmulq_s32(input4, kernel4));
                        } else {
                            for (size_t c = 0; c < bytesPerPixel - 1; c++) {
                                value[c] += inputBuffer[inputIndex + c] * filter.kernel[kernelIndex];
                            }
                        }
                    }
                }
                size_t outputIndex = y_head + (x * bytesPerPixel);
                if (neon) {
#ifdef __ARM_NEON__
                    float32x4_t value4f = {
                        (float32_t)vgetq_lane_s32(value4, 0),
                        (float32_t)vgetq_lane_s32(value4, 1),
                        (float32_t)vgetq_lane_s32(value4, 2),
                        (float32_t)vgetq_lane_s32(value4, 3)
                    };
                    value4f = vabsq_f32(vdivq_f32(value4f, vdupq_n_f32((float32_t)filter.divisor)));
                    value4f = vminq_f32(value4f, vdupq_n_f32((float32_t)UCHAR_MAX));
                    value4f = vdivq_f32(value4f, vdupq_n_f32((float32_t)filter.divisor));
//                    value4f = vminq_f32(value4f, vdupq_n_f32((float32_t)UCHAR_MAX));
//                    value4f = vmaxq_f32(value4f, vdupq_n_f32(0.f));
//                    outputBuffer[outputIndex + 0] = (uint8_t)vgetq_lane_f32(value4f, 0);
                    outputBuffer[outputIndex + 1] = (uint8_t)vgetq_lane_f32(value4f, 1);
                    outputBuffer[outputIndex + 2] = (uint8_t)vgetq_lane_f32(value4f, 2);
#endif
                } else {
                    for (size_t c = 0; c < bytesPerPixel - 1; c++) {
                        value[c] /= filter.divisor;
                        value[c] = value[c] < UCHAR_MAX ? value[c] : UCHAR_MAX;
                        value[c] = value[c] < 0 ? 0 : value[c];
                        outputBuffer[outputIndex + c] = (uint8_t)value[c];
                    }
                }
            }
        }
    });
    
    free(begin);
    free(end);
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef cgContext = CGBitmapContextCreate(outputBuffer, width, height, 8,
                                                   bytesPerRow, colorSpace, bitmapInfo);
    CGImageRef outputImage = CGBitmapContextCreateImage(cgContext);
    free(outputBuffer);
    
    CGContextRelease(cgContext);
    CGColorSpaceRelease(colorSpace);
    
    return outputImage;
}

CGImageRef CreateFilteredImage_GPU_vImage(CVImageBufferRef buffer, ConvolutionFilter filter) {
    const OSType type = CVPixelBufferGetPixelFormatType(buffer);
    const size_t width  = CVPixelBufferGetWidth(buffer);
    const size_t height = CVPixelBufferGetHeight(buffer);
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
    const size_t bufferSize = CVPixelBufferGetDataSize(buffer);
    if (type != kCVPixelFormatType_32BGRA) return NULL;
   
    const vImage_Buffer inputBuffer = {
        CVPixelBufferGetBaseAddress(buffer), height, width, bytesPerRow};
    
    const vImage_Buffer outputBuffer = {
        malloc(bufferSize), height, width, bytesPerRow};
    
    Pixel_8888 backgroundColor = {0,0,0,0};
    vImageConvolve_ARGB8888(&inputBuffer, &outputBuffer, NULL, 0, 0,
                            filter.kernel, filter.size, filter.size, filter.divisor,
                            backgroundColor, kvImageCopyInPlace);
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef cgContext = CGBitmapContextCreate(outputBuffer.data, width, height, 8,
                                                   bytesPerRow, colorSpace, bitmapInfo);
    CGImageRef outputImage = CGBitmapContextCreateImage(cgContext);
    free(outputBuffer.data);
    
    CGContextRelease(cgContext);
    CGColorSpaceRelease(colorSpace);
    
    return outputImage;
}

