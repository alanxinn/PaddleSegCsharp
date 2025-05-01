using OpenCvSharp;
using PaddleSegCsharp;
using Sdcb.PaddleInference;
using System;
using System.Collections.Generic;

SegModel segModel = SegModel.FromDirectory(
    @"inference model folder");

PaddleSegPredictor paddleSegPredictor = new PaddleSegPredictor(segModel, PaddleDevice.Mkldnn());

Mat img = new Mat(@"test image path");
string outputPath = "test_segmented.png";

Console.WriteLine("Press any key to start inference");
Console.ReadKey();

var result = paddleSegPredictor.Run(img);

// Process segmentation results and apply color filling
Mat segmentationImage = new Mat(img.Size(), MatType.CV_8UC3);
Dictionary<int, Scalar> classColors = new Dictionary<int, Scalar>
{
    // Define colors for each of the 19 classes
    {0, new Scalar(0, 0, 0)},         // Class 0 - Black (Background)
    {1, new Scalar(128, 0, 0)},       // Class 1 - Maroon
    {2, new Scalar(0, 128, 0)},       // Class 2 - Green
    {3, new Scalar(128, 128, 0)},     // Class 3 - Olive
    {4, new Scalar(0, 0, 128)},       // Class 4 - Navy
    {5, new Scalar(128, 0, 128)},     // Class 5 - Purple
    {6, new Scalar(0, 128, 128)},     // Class 6 - Teal
    {7, new Scalar(128, 128, 128)},   // Class 7 - Gray
    {8, new Scalar(64, 0, 0)},        // Class 8 - Dark Red
    {9, new Scalar(192, 0, 0)},       // Class 9 - Red
    {10, new Scalar(64, 128, 0)},     // Class 10 - Dark Yellow-Green
    {11, new Scalar(192, 128, 0)},    // Class 11 - Orange
    {12, new Scalar(64, 0, 128)},     // Class 12 - Dark Purple-Blue
    {13, new Scalar(192, 0, 128)},    // Class 13 - Pink
    {14, new Scalar(64, 128, 128)},   // Class 14 - Dark Cyan
    {15, new Scalar(192, 128, 128)},  // Class 15 - Light Pink
    {16, new Scalar(0, 64, 0)},       // Class 16 - Dark Green
    {17, new Scalar(128, 64, 0)},     // Class 17 - Brown
    {18, new Scalar(0, 192, 0)}       // Class 18 - Bright Green
};

// Map segmentation results to color image
int width = img.Width;
int height = img.Height;
for (int y = 0; y < height; y++)
{
    for (int x = 0; x < width; x++)
    {
        int index = y * width + x;
        if (index < result.Length)
        {
            int classId = result[index];
            if (classColors.ContainsKey(classId))
            {
                segmentationImage.Set(y, x, classColors[classId]);
            }
            else
            {
                // If color is not defined for the class, use white
                segmentationImage.Set(y, x, new Scalar(255, 255, 255));
            }
        }
    }
}

// Create semi-transparent overlay
Mat original = img.Clone();
Cv2.CvtColor(original, original, ColorConversionCodes.BGR2RGB); // Ensure original image is in RGB format
Mat blended = new Mat();
Cv2.AddWeighted(original, 0.7, segmentationImage, 0.3, 0, blended);

// Save the result
Cv2.ImWrite(outputPath, blended);

Console.WriteLine($"Segmentation result saved to: {outputPath}");
Console.WriteLine("Press any key to exit");

Console.ReadKey();