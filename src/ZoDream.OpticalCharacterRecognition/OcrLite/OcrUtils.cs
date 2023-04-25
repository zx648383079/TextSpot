using Emgu.CV;
using Emgu.CV.CvEnum;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZoDream.OpticalCharacterRecognition.OcrLite
{
    public static class OcrUtils
    {
        public static Tensor<float> SubstractMeanNormalize(Bitmap src, float[] meanVals, float[] normVals)
        {
            int cols = src.Width;
            int rows = src.Height;
            int channels = 3;
            // byte[,,] imgData = srcImg.Data;
            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, channels, rows, cols });
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    var color = src.GetPixel(c, r);
                    for (var ch = 0; ch < channels; ch++)
                    {
                        var value = ch switch { 
                            1 => color.G,
                            2 => color.B,
                            _ => color.R,
                        };
                        float data = (float)(value * normVals[ch] - meanVals[ch] * normVals[ch]);
                        inputTensor[0, ch, r, c] = data;
                    }
                }
            }
            return inputTensor;
        }
        /// <summary>
        /// 给图像边缘增加区域
        /// </summary>
        /// <param name="src"></param>
        /// <param name="padding"></param>
        /// <returns></returns>
        public static Bitmap MakePadding(Bitmap src, int padding)
        {
            if (padding <= 0)
            {
                return src;
            }
            var paddingSrc = new Bitmap(src.Width + padding * 2, src.Height + padding * 2, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(paddingSrc);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.White);
            g.DrawImage(src, new Point(padding, padding));
            g.Save();
            src.Dispose();
            return paddingSrc;
        }

        public static Bitmap Resize(Bitmap src, int width, int height)
        {
            var dst = new Bitmap(width, height);
            using var g = Graphics.FromImage(dst);
            g.DrawImage(src, 0,0, width,height);
            g.Save();
            return dst;
        }

        public static int GetThickness(Bitmap boxImg)
        {
            int minSize = boxImg.Width > boxImg.Height ? boxImg.Height : boxImg.Width;
            int thickness = minSize / 1000 + 2;
            return thickness;
        }

        public static void DrawTextBox(Bitmap boxImg, List<Point> box, int thickness)
        {
            if (box == null || box.Count == 0)
            {
                return;
            }
            var pen = new Pen(Color.Red, thickness);
            using var g = Graphics.FromImage(boxImg);
            g.DrawLine(pen, box[0], box[1]);
            g.DrawLine(pen, box[1], box[2]);
            g.DrawLine(pen, box[2], box[3]);
            g.DrawLine(pen, box[3], box[0]);
            g.Save();
        }

        public static void DrawTextBoxes(Bitmap src, List<TextBox> textBoxes, int thickness)
        {
            for (int i = 0; i < textBoxes.Count; i++)
            {
                TextBox t = textBoxes[i];
                DrawTextBox(src, t.Points, thickness);
            }
        }

        public static List<Bitmap> GetPartImages(Bitmap src, List<TextBox> textBoxes)
        {
            var partImages = new List<Bitmap>();
            for (int i = 0; i < textBoxes.Count; ++i)
            {
                var partImg = GetRotateCropImage(src, textBoxes[i].Points);
                //Mat partImg = new Mat();
                //GetRoiFromBox(src, partImg, textBoxes[i].Points);
                partImages.Add(partImg);
            }
            return partImages;
        }

        public static Bitmap CopyPart(Bitmap src, Rectangle rect)
        {
            var dst = new Bitmap(rect.Width, rect.Height);
            using var g = Graphics.FromImage(dst);
            g.DrawImage(src, 
                new Rectangle(0, 0, rect.Width, rect.Height), rect, GraphicsUnit.Pixel);
            g.Save();
            return dst;
        }

        public static Bitmap GetRotateCropImage(Bitmap src, List<Point> box)
        {
            var image = new Bitmap(src);
            var points = new List<Point>();
            points.AddRange(box);

            int[] collectX = { box[0].X, box[1].X, box[2].X, box[3].X };
            int[] collectY = { box[0].Y, box[1].Y, box[2].Y, box[3].Y };
            int left = collectX.Min();
            int right = collectX.Max();
            int top = collectY.Min();
            int bottom = collectY.Max();

            var rect = new Rectangle(left, top, right - left, bottom - top);
            var imgCrop = CopyPart(src, rect);

            for (int i = 0; i < points.Count; i++)
            {
                var pt = points[i];
                pt.X -= left;
                pt.Y -= top;
                points[i] = pt;
            }

            int imgCropWidth = (int)(Math.Sqrt(Math.Pow(points[0].X - points[1].X, 2) +
                                        Math.Pow(points[0].Y - points[1].Y, 2)));
            int imgCropHeight = (int)(Math.Sqrt(Math.Pow(points[0].X - points[3].X, 2) +
                                         Math.Pow(points[0].Y - points[3].Y, 2)));

            var ptsDst0 = new PointF(0, 0);
            var ptsDst1 = new PointF(imgCropWidth, 0);
            var ptsDst2 = new PointF(imgCropWidth, imgCropHeight);
            var ptsDst3 = new PointF(0, imgCropHeight);

            PointF[] ptsDst = { ptsDst0, ptsDst1, ptsDst2, ptsDst3 };


            var ptsSrc0 = new PointF(points[0].X, points[0].Y);
            var ptsSrc1 = new PointF(points[1].X, points[1].Y);
            var ptsSrc2 = new PointF(points[2].X, points[2].Y);
            var ptsSrc3 = new PointF(points[3].X, points[3].Y);

            PointF[] ptsSrc = { ptsSrc0, ptsSrc1, ptsSrc2, ptsSrc3 };
            // 透视转化
            Mat M = CvInvoke.GetPerspectiveTransform(ptsSrc, ptsDst);

            var partImg = new Mat();
            // 对图像进行透视变换，就是变形
            CvInvoke.WarpPerspective(imgCrop.ToMat(), partImg, M,
                                new Size(imgCropWidth, imgCropHeight), Inter.Nearest, Warp.Default,
                               BorderType.Replicate);

            if (partImg.Height >= partImg.Width * 1.5)
            {
                var srcCopy = new Mat();
                // 转置，相当于沿对角线翻转
                CvInvoke.Transpose(partImg, srcCopy);
                // 垂直翻转图像
                CvInvoke.Flip(srcCopy, srcCopy, 0);
                return srcCopy.ToBitmap();
            }
            else
            {
                return partImg.ToBitmap();
            }
        }

        /// <summary>
        /// 垂直翻转加水平翻转
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Bitmap MatRotateClockWise180(Bitmap src)
        {
            src.RotateFlip(RotateFlipType.Rotate180FlipNone);
            //CvInvoke.Flip(src, src, FlipType.Vertical);
            //CvInvoke.Flip(src, src, FlipType.Horizontal);
            return src;
        }

        /// <summary>
        /// 逆时针旋转90
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Bitmap MatRotateClockWise90(Bitmap src)
        {
            src.RotateFlip(RotateFlipType.Rotate270FlipXY);
            return src;
        }
    }
}
