//
//  ImageFilteringByOpenCV.h
//  ImageFiltering
//
//  Created by temoki on 2015/01/13.
//  Copyright (c) 2015年 temoki. All rights reserved.
//

#ifndef __ImageFiltering__ImageFilteringByOpenCV__
#define __ImageFiltering__ImageFilteringByOpenCV__

#include <CoreGraphics/CoreGraphics.h>
#include <CoreVideo/CoreVideo.h>

CGImageRef CreateOpenCVGaussianFilteredImage(CVImageBufferRef buffer, int kernelSize);

#endif /* defined(__ImageFiltering__ImageFilteringByOpenCV__) */
