using Sdcb.PaddleInference;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace PaddleSegCsharp
{
    internal class PaddleSegPredictor : IDisposable
    {
        private readonly PaddlePredictor _p;

        public PaddleSegPredictor(SegModel model, Action<PaddleConfig>? configure = null)
        {
            PaddleConfig c = model.CreateConfig();
            model.ConfigureDevice(c, configure);

            _p = c.CreatePredictor();
        }

        public float[] Run(Mat img)
        {
            Cv2.CvtColor(img, img, ColorConversionCodes.BGR2RGB);
            Mat mat = Normalize(img);
            using PaddlePredictor predictor = _p.Clone();
            using (PaddleTensor input = predictor.GetInputTensor(predictor.InputNames[0]))
            {
                input.Shape = new[] { 1, 3, mat.Rows, mat.Cols };
                float[] data = ExtractMat(mat);
                input.SetData(data);
            }
            if (!predictor.Run())
            {
                throw new Exception("PaddlePredictor(Detector) run failed.");
            }
            using (PaddleTensor output = predictor.GetOutputTensor(predictor.OutputNames[0]))
            {
                float[] data = output.GetData<float>();
                int[] shape = output.Shape;
                return data;
            }
        }

        private static Mat Normalize(Mat src)
        {
            Mat imFloat = new Mat();
            src.ConvertTo(imFloat, MatType.CV_32FC3, 1.0 / 255.0);

            imFloat = (imFloat - 0.5) / 0.5;

            return imFloat;
        }

        private static float[] ExtractMat(Mat src)
        {
            int rows = src.Rows;
            int cols = src.Cols;
            float[] result = new float[rows * cols * 3];
            GCHandle resultHandle = default;
            try
            {
                resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
                IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
                for (int i = 0; i < src.Channels(); ++i)
                {
                    int rgb = 2 - i;
                    using Mat dest = new(rows, cols, MatType.CV_32FC1, resultPtr + i * rows * cols * sizeof(float));
                    Cv2.ExtractChannel(src, dest, rgb);
                }
            }
            finally
            {
                resultHandle.Free();
            }
            return result;
        }

        public void Dispose()
        {
            _p.Dispose();
        }
    }
}