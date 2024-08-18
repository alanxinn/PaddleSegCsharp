// See https://aka.ms/new-console-template for more information
using OpenCvSharp;
using PaddleSegCsharp;
using Sdcb.PaddleInference;
using System.Diagnostics;

Console.WriteLine("Hello, World!");

void PaddleSegInfer()
{
    // Mat image = new Mat(@"E:\desktop\0822Cam1baiquan1.bmp");
    // // image.Resize(new Size(1024, 1024));
    // Cv2.Resize(image, image, new Size(128, 128));
    // Cv2.ImWrite(@"E:\desktop\resize128.bmp", image);

    SegModel segModel = SegModel.FromDirectory(
        @"");

    PaddleSegPredictor paddleSegPredictor = new PaddleSegPredictor(segModel, PaddleDevice.Gpu());

    Mat img = new Mat(@"");

    Stopwatch sw = new Stopwatch();
    double time = 0;
    int count = 1;
    Console.WriteLine("按下任意键开始推理");
    Console.ReadKey();

    for (int j = 0; j < count; j++)
    {
        sw.Restart();
        float[] result = paddleSegPredictor.Run(img);
        sw.Stop();
        byte[] res1 = new byte[262144];

        for (int i = 262144; i < 524288; i++)
        {
            if (result[i] >= 0.6)
            {
                res1[i - 262144] = 255;
            }
        }

        Mat mat = new Mat(512, 512, MatType.CV_8UC1, res1);

        Cv2.FindContours(mat, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        Cv2.DrawContours(img, contours, -1, new Scalar(0, 0, 255), 1);

        Cv2.ImWrite(@"", img);

        Mat resMat = new Mat();
        Cv2.Resize(img, resMat, new Size(1440, 1080));
        Cv2.ImWrite(@"", resMat);
    }

    Console.WriteLine($"{count}次平均花费时间为：{time / count} ms");

    Console.WriteLine("按下任意键结束");

    Console.ReadKey();
}