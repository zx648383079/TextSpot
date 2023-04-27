using Emgu.CV.Ocl;
using System;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZoDream.OpticalCharacterRecognition.OcrNet
{
    public static class FontDataGenerator
    {

        public static void Generate(FontDataLoader loader, string inputFolder)
        {
            var font = new Font(loader.Font, 36);
            var fontBrush = new SolidBrush(Color.Black);
            var format = new StringFormat
            {
                Alignment = StringAlignment.Center,
                LineAlignment = StringAlignment.Center
            };
            var width = 64;
            var height = 64;
            var rect = new RectangleF(0, 0, width, height);
            MakeFolder(inputFolder);
            foreach (var item in loader)
            {
                var img = new Bitmap(width, height);
                using var g = Graphics.FromImage(img);
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.Clear(Color.White);
                g.DrawString(FontDataLoader.ToChar(item), font, fontBrush, rect, format);
                img.Save(Path.Combine(MakeFolder(Path.Combine(inputFolder, 
                    item.ToString())), $"{font.Name}.jpg"),
                    ImageFormat.Jpeg);
                img.Dispose();
            }
        }

        private static string MakeFolder(string file)
        {
            if (Directory.Exists(file))
            {
                return file;
            }
            Directory.CreateDirectory(file);
            return file;
        }
    }
}
