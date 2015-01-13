//
//  ImageFilteringByOpenCV.cpp
//  ImageFiltering
//
//  Created by temoki on 2015/01/13.
//  Copyright (c) 2015年 temoki. All rights reserved.
//

#include "ImageFilteringByOpenCV.h"
#include "ImageFiltering.h"

// OpenCV
#include <opencv2/opencv.hpp>

CGImageRef CreateOpenCVGaussianFilteredImage(CVImageBufferRef buffer, int kernelSize) {
    const OSType type = CVPixelBufferGetPixelFormatType(buffer);
    if (type != kCVPixelFormatType_32BGRA) return NULL;

    // CVImageBufferRef -> cv::Mat
    const int width  = static_cast<int>(CVPixelBufferGetWidth(buffer));
    const int height = static_cast<int>(CVPixelBufferGetHeight(buffer));
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);
    void * baseAddress = CVPixelBufferGetBaseAddress(buffer);
    cv::Mat srcMat = cv::Mat(height, width, CV_8UC4, baseAddress);

    // Gaussian Filter
    cv::Mat dstMat = cv::Mat(height, width, CV_8UC4);
    cv::Size size(kernelSize, kernelSize);
    cv::GaussianBlur(srcMat, dstMat, size, 1.f);

    // cv::Mat -> CGImageRef
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef cgContext = CGBitmapContextCreate(srcMat.data, width, height, 8,
                                                   bytesPerRow, colorSpace, bitmapInfo);
    CGImageRef outputImage = CGBitmapContextCreateImage(cgContext);
    CGContextRelease(cgContext);
    CGColorSpaceRelease(colorSpace);
    
    return outputImage;
}
