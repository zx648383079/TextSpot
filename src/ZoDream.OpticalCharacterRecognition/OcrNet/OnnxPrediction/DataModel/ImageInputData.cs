using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace ZoDream.OpticalCharacterRecognition.OcrNet.OnnxPrediction.DataModel
{
    public struct ImageSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }

    public class ImageInputData
    {
        [ImageType(ImageSettings.imageHeight, ImageSettings.imageWidth)]
        public Bitmap Image { get; set; }
    }
}
