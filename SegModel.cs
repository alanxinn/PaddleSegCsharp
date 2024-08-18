using Sdcb.PaddleInference;

namespace PaddleSegCsharp
{
    public abstract class SegModel : BaseModel
    {
        public static SegModel FromDirectory(string directoryPath) => new FileSegModel(directoryPath);

        public override Action<PaddleConfig> DefaultDevice => PaddleDevice.Mkldnn();
    }
}