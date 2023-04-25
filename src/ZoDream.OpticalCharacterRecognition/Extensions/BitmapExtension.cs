using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Media.Imaging;
using System.Windows.Media;

namespace ZoDream.Studio.Extensions
{
    public static class BitmapExtension
    {
        private static BitmapPalette? MyBitmapPalette;

        public static BitmapSource ToBitmapSource(this Bitmap bitmap)
        {
            if (MyBitmapPalette is null)
            {
                var colors = new List<System.Windows.Media.Color>
                {
                    Colors.Red,
                    Colors.Blue,
                    Colors.Green
                };
                MyBitmapPalette = new BitmapPalette(colors);
            }
            var bmpData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            var bitmapSource = BitmapSource.Create(bitmap.Width, bitmap.Height, 96, 96, PixelFormats.Bgr24, MyBitmapPalette, bmpData.Scan0, bitmap.Width * bitmap.Height * 3, bitmap.Width * 3);
            bitmap.UnlockBits(bmpData);
            return bitmapSource;
        }
    }
}
