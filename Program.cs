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
    // Define colors for each class, adjust the number of classes and colors based on your model
    {0, new Scalar(0, 0, 0)},     // Background - Black
    {1, new Scalar(0, 0, 255)},   // Class 1 - Red
    {2, new Scalar(0, 255, 0)},   // Class 2 - Green
    {3, new Scalar(255, 0, 0)},   // Class 3 - Blue
    // Add more classes and colors...
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
                // If color is not defined for the class, use purple
                segmentationImage.Set(y, x, new Scalar(255, 0, 255));
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