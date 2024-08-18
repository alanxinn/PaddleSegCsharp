using OpenCvSharp;
using PaddleSegCsharp;
using Sdcb.PaddleInference;

SegModel segModel = SegModel.FromDirectory(
    @"inference model folder");

PaddleSegPredictor paddleSegPredictor = new PaddleSegPredictor(segModel, PaddleDevice.Mkldnn());

Mat img = new Mat(@"test image path");

Console.WriteLine("按下任意键开始推理");
Console.ReadKey();

float[] result = paddleSegPredictor.Run(img);

Console.WriteLine("按下任意键结束");

Console.ReadKey();