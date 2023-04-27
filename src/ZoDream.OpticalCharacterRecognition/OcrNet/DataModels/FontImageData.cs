using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZoDream.OpticalCharacterRecognition.OcrNet.DataModels
{
    public class FontImageData
    {
        public FontImageData(string imagePath, string label)
        {
            ImagePath = imagePath;
            Label = label;
        }

        public readonly string ImagePath;

        public readonly string Label;
    }
}
