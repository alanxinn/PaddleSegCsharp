using OpenCvSharp;
using PaddleSegCsharp;
using Sdcb.PaddleInference;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

SegModel segModel = SegModel.FromDirectory(
    @"inference model folder");

PaddleSegPredictor paddleSegPredictor = new PaddleSegPredictor(segModel, PaddleDevice.Mkldnn());

Mat img = new Mat(@"test image path");
string outputPath = "test_segmented.png";
string resultDataPath = "segmentation_results.json";
string resultRawPath = "segmentation_raw.bin";

Console.WriteLine("Press any key to start inference");
Console.ReadKey();

var result = paddleSegPredictor.Run(img);

// Save raw segmentation results to binary file
using (FileStream fs = new FileStream(resultRawPath, FileMode.Create))
{
    byte[] resultBytes = new byte[result.Length * sizeof(int)];
    Buffer.BlockCopy(result, 0, resultBytes, 0, resultBytes.Length);
    fs.Write(resultBytes, 0, resultBytes.Length);
}

// Save segmentation results with metadata to JSON
var resultData = new
{
    Width = img.Width,
    Height = img.Height,
    ClassCount = 19,
    Timestamp = DateTime.Now,
    PixelClassifications = result
};

string jsonString = JsonSerializer.Serialize(resultData, new JsonSerializerOptions { WriteIndented = true });
File.WriteAllText(resultDataPath, jsonString);

Console.WriteLine($"Raw segmentation data saved to: {resultRawPath}");
Console.WriteLine($"Segmentation data with metadata saved to: {resultDataPath}");

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

// Also save a raw class map image (grayscale representation of classes)
Mat classMapImage = new Mat(img.Size(), MatType.CV_8UC1);

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
            
            // Save class ID to grayscale image (clamped to 0-255)
            classMapImage.Set(y, x, (byte)Math.Min(classId, 255));
            
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

// Save the class map (grayscale)
Cv2.ImWrite("class_map.png", classMapImage);
Console.WriteLine("Class map saved to: class_map.png");

// Create semi-transparent overlay
Mat original = img.Clone();
Cv2.CvtColor(original, original, ColorConversionCodes.BGR2RGB); // Ensure original image is in RGB format
Mat blended = new Mat();
Cv2.AddWeighted(original, 0.7, segmentationImage, 0.3, 0, blended);

// Save the visualization result
Cv2.ImWrite(outputPath, blended);

// Save the color-filled segmentation image without blending
Cv2.ImWrite("segmentation_colors.png", segmentationImage);
Console.WriteLine("Color segmentation image saved to: segmentation_colors.png");

Console.WriteLine($"Visualization result saved to: {outputPath}");
Console.WriteLine("Press any key to exit");

Console.ReadKey();