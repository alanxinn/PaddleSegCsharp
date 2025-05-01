# PaddleSegCsharp

PaddleSegCsharp is a C# application for image segmentation using PaddlePaddle inference models. The project provides an easy-to-use interface for running semantic segmentation models and visualizing results.

## Features

- Load PaddlePaddle segmentation models
- Run inference on images
- Visualize segmentation results with color overlays
- Customizable class-to-color mapping

## Requirements

- .NET 6.0 or higher
- OpenCvSharp4
- Sdcb.PaddleInference

## Getting Started

1. Clone the repository
2. Place your PaddlePaddle segmentation model in a directory
3. Update the model path in `Program.cs`:
   ```csharp
   SegModel segModel = SegModel.FromDirectory(@"path/to/your/model");
   ```
4. Update the test image path:
   ```csharp
   Mat img = new Mat(@"path/to/your/image.jpg");
   ```
5. Run the application:
   ```
   dotnet run
   ```

## Code Structure

- `Program.cs`: Main application entry point and segmentation visualization
- `PaddleSegPredictor.cs`: Wrapper for PaddlePaddle inference
- `SegModel.cs`: Model loader and configuration
- `BaseModel.cs`: Base abstractions for models

## Customizing Segmentation Visualization

You can customize the visualization by modifying the color mapping in `Program.cs`:

```csharp
Dictionary<int, Scalar> classColors = new Dictionary<int, Scalar>
{
    {0, new Scalar(0, 0, 0)},     // Background - Black
    {1, new Scalar(0, 0, 255)},   // Class 1 - Red
    {2, new Scalar(0, 255, 0)},   // Class 2 - Green
    {3, new Scalar(255, 0, 0)},   // Class 3 - Blue
    // Add more classes and colors...
};
```

You can also adjust the transparency of the overlay by changing the weights:

```csharp
Cv2.AddWeighted(original, 0.7, segmentationImage, 0.3, 0, blended);
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2023 PaddleSegCsharp Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgements

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
- [OpenCvSharp](https://github.com/shimat/opencvsharp)
- [Sdcb.PaddleInference](https://github.com/sdcb/PaddleSharp) 