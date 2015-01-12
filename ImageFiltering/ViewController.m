//
//  ViewController.m
//  ImageFiltering
//
//  Created by temoki on 2015/01/07.
//  Copyright (c) 2015年 temoki. All rights reserved.
//

#import "ViewController.h"
#import "ImageFiltering.h"
#import <Accelerate/Accelerate.h>
#import <arm_neon.h>

@interface ViewController ()
- (IBAction)onFilterSizeSegment:(id)sender;
- (IBAction)onOperationSegmentChanged:(id)sender;
- (CGImageRef)createGaussianFilteredImage_GPU_CoreImage:(CVImageBufferRef)imageBuffer radius:(float)radius;
@end

@implementation ViewController {
    IBOutlet UIView *               previewView;
    IBOutlet UISegmentedControl *   filterSizeSegment;
    IBOutlet UISlider *             radiusSlider;
    IBOutlet UILabel *              infoLabel;
    IBOutlet UISegmentedControl *   operationSegment;
    
    AVCaptureSession *              captureSession;
    AVCaptureDeviceInput *          captureInput;
    AVCaptureVideoDataOutput *      captureOutput;
    AVCaptureVideoPreviewLayer *    previewLayer;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Setup UI Controls
    filterSizeSegment.selectedSegmentIndex = 3; // 7x7
    operationSegment.selectedSegmentIndex = 0;  // CPU
    radiusSlider.minimumValue = 0.f;
    radiusSlider.maximumValue = 10.f;
    radiusSlider.value = 1.f;
    radiusSlider.hidden = YES;
    
    // Setup capture input
    AVCaptureDevice * device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    captureInput = [AVCaptureDeviceInput deviceInputWithDevice:device error:NULL];
    
    // Setup capture output
    NSDictionary * settings = @{(NSString *)kCVPixelBufferPixelFormatTypeKey:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA]};
    captureOutput = [AVCaptureVideoDataOutput new];
    captureOutput.videoSettings = settings;
    captureOutput.alwaysDiscardsLateVideoFrames = YES;
    dispatch_queue_t queue = dispatch_queue_create("CaptureQueue", DISPATCH_QUEUE_SERIAL);
    [captureOutput setSampleBufferDelegate:self queue:queue];
    
    // Setup capture session
    captureSession = [AVCaptureSession new];
    [captureSession addInput:captureInput];
    [captureSession addOutput:captureOutput];
    [captureSession beginConfiguration];
    captureSession.sessionPreset = AVCaptureSessionPresetMedium;
    for (AVCaptureConnection *connection in [captureOutput connections]) {
        for (AVCaptureInputPort *port in [connection inputPorts]) {
            if ([[port mediaType] isEqual:AVMediaTypeVideo] && [connection isVideoOrientationSupported]) {
                [connection setVideoOrientation:AVCaptureVideoOrientationPortrait];
                break;
            }
        }
    }
    [captureSession commitConfiguration];
    
    // Add video preview layer
    previewLayer = [AVCaptureVideoPreviewLayer layer];
    previewLayer.frame = self.view.bounds;
    previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    [previewView.layer addSublayer:previewLayer];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)viewDidAppear:(BOOL)animated {
    [captureSession startRunning];
}

- (void)viewDidDisappear:(BOOL)animated {
    [captureSession stopRunning];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    const size_t width  = CVPixelBufferGetWidth(imageBuffer);
    const size_t height = CVPixelBufferGetHeight(imageBuffer);
    const size_t numOfCpu = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
    
    CGImageRef outputImage = NULL;
    NSString * processText = nil;
    clock_t start = clock();
    if (filterSizeSegment.selectedSegmentIndex == 0) {
        processText = @"Normal";
        outputImage = CreateImage(imageBuffer);
    } else {
        ConvolutionFilter gaussianFilter;
        switch (filterSizeSegment.selectedSegmentIndex) {
            case 1:
                gaussianFilter = GaussianFilter3x3();
                break;
            case 2:
                gaussianFilter = GaussianFilter5x5();
                break;
            case 3:
            default:
                gaussianFilter = GaussianFilter7x7();
                break;
        }

        switch (operationSegment.selectedSegmentIndex) {
            case 1:
                processText = @"CPU(Neon)";
                outputImage = CreateFilteredImage_CPU(imageBuffer, gaussianFilter, true);
                break;
            case 2:
                processText = [NSString stringWithFormat:@"CPU(%lu Threads)", numOfCpu];
                outputImage = CreateFilteredImage_CPU_MultiThread(imageBuffer, gaussianFilter, false, numOfCpu);
                break;
            case 3:
                processText = @"GPU(vImage)";
                outputImage = CreateFilteredImage_GPU_vImage(imageBuffer, gaussianFilter);
                break;
            case 4:
                processText = [NSString stringWithFormat:@"GPU(CIFilter, radius=%2.1f)", radiusSlider.value];
                outputImage = [self createGaussianFilteredImage_GPU_CoreImage:imageBuffer radius:radiusSlider.value];
                break;
            default:
                processText = @"CPU";
                outputImage = CreateFilteredImage_CPU(imageBuffer, gaussianFilter, false);
        }
    }
    float elapseTime = (clock() - start) / (float)CLOCKS_PER_SEC;
    NSString * elapseTimeText = @"###.###fps";
    if (elapseTime) {
        elapseTimeText = [NSString stringWithFormat:@"%3.3ffps", (1.f / elapseTime)];
    }
    NSString *labelText = [NSString stringWithFormat:@"[%@] Size[%lu,%lu] [%@]", processText, width, height, elapseTimeText];
    
    if (outputImage) {
        dispatch_sync(dispatch_get_main_queue(), ^{
            infoLabel.text = labelText;
            previewLayer.contents = (__bridge id)outputImage;
        });
        CGImageRelease(outputImage);
    }
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
}

- (IBAction)onFilterSizeSegment:(id)sender {
    BOOL enabled = (filterSizeSegment.selectedSegmentIndex == 0)? NO : YES;
    radiusSlider.enabled = enabled;
    operationSegment.enabled = enabled;
}

- (IBAction)onOperationSegmentChanged:(id)sender {
    radiusSlider.hidden = (operationSegment.selectedSegmentIndex == 4)? NO : YES;
}

- (CGImageRef)createGaussianFilteredImage_GPU_CoreImage:(CVImageBufferRef)imageBuffer radius:(float)radius {
    CIImage * inputImage = [CIImage imageWithCVPixelBuffer:imageBuffer];
    CIFilter * filter = [CIFilter filterWithName:@"CIGaussianBlur"];
    [filter setValue:inputImage forKey:@"inputImage"];
    [filter setValue:[NSNumber numberWithFloat:radius] forKey:@"inputRadius"];

    CIImage * filteredImage = filter.outputImage;
    
    CIContext *ciContext = [CIContext contextWithOptions:nil];
    CGImageRef outputImage = [ciContext createCGImage:filteredImage
                                          fromRect:[filteredImage extent]];

    return outputImage;
}

@end
