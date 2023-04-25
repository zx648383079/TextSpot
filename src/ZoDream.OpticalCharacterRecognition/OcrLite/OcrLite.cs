using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZoDream.OpticalCharacterRecognition.OcrLite
{
    public class OcrLite
    {
        public bool IsPartImg { get; set; }
        public bool IsDebugImg { get; set; }
        private readonly DbNet _dbNet = new();
        private readonly AngleNet _angleNet = new();
        private readonly CrnnNet _crnnNet = new();

        public void InitModels(string detPath, string clsPath, string recPath, string keysPath, int numThread)
        {
            try
            {
                _dbNet.InitModel(detPath, numThread);
                _angleNet.InitModel(clsPath, numThread);
                _crnnNet.InitModel(recPath, keysPath, numThread);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public OcrResult Detect(string img, int padding, int imgResize, float boxScoreThresh, float boxThresh,
                              float unClipRatio, bool doAngle, bool mostAngle)
        {
            var originSrc = (Bitmap)Image.FromFile(img, true);//default : BGR
            var originRect = new Rectangle(padding, padding, originSrc.Width, originSrc.Height);
            var paddingSrc = OcrUtils.MakePadding(originSrc, padding);

            int resize;
            if (imgResize <= 0)
            {
                resize = Math.Max(paddingSrc.Width, paddingSrc.Height);
            }
            else
            {
                resize = imgResize;
            }
            var scale = ScaleParam.GetScaleParam(paddingSrc, resize);

            return DetectOnce(paddingSrc, originRect, scale, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
        }

        public OcrResult Detect(Bitmap originSrc, int padding, int imgResize, float boxScoreThresh, float boxThresh,
                              float unClipRatio, bool doAngle, bool mostAngle)
        {
            var originRect = new Rectangle(padding, padding, originSrc.Width, originSrc.Height);
            var paddingSrc = OcrUtils.MakePadding(originSrc, padding);

            int resize;
            if (imgResize <= 0)
            {
                resize = Math.Max(paddingSrc.Width, paddingSrc.Height);
            }
            else
            {
                resize = imgResize;
            }
            var scale = ScaleParam.GetScaleParam(paddingSrc, resize);

            return DetectOnce(paddingSrc, originRect, scale, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
        }

        private OcrResult DetectOnce(Bitmap src, Rectangle originRect, ScaleParam scale, float boxScoreThresh, float boxThresh,
                              float unClipRatio, bool doAngle, bool mostAngle)
        {
            var textBoxPaddingImg = (Bitmap)src.Clone();
            int thickness = OcrUtils.GetThickness(src);
            Console.WriteLine("=====Start detect=====");
            var watcher = new Stopwatch();
            watcher.Start();

            Console.WriteLine("---------- step: dbNet getTextBoxes ----------");
            var textBoxes = _dbNet.GetTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
            watcher.Stop();
            var dbNetTime = watcher.ElapsedMilliseconds;

            Console.WriteLine($"TextBoxesSize({textBoxes.Count})");
            textBoxes.ForEach(x => Console.WriteLine(x));
            //Console.WriteLine($"dbNetTime({dbNetTime}ms)");

            watcher.Start();
            Console.WriteLine("---------- step: drawTextBoxes ----------");
            OcrUtils.DrawTextBoxes(textBoxPaddingImg, textBoxes, thickness);
            //CvInvoke.Imshow("ResultPadding", textBoxPaddingImg);

            //---------- getPartImages ----------
            var partImages = OcrUtils.GetPartImages(src, textBoxes);
            if (IsPartImg)
            {
                for (int i = 0; i < partImages.Count; i++)
                {
                    // CvInvoke.Imshow($"PartImg({i})", partImages[i]);
                }
            }

            Console.WriteLine("---------- step: angleNet getAngles ----------");
            var angles = _angleNet.GetAngles(partImages, doAngle, mostAngle);
            //angles.ForEach(x => Console.WriteLine(x));

            //Rotate partImgs
            for (int i = 0; i < partImages.Count; ++i)
            {
                if (angles[i].Index == 0)
                {
                    partImages[i] = OcrUtils.MatRotateClockWise180(partImages[i]);
                }
                if (IsDebugImg)
                {
                    // CvInvoke.Imshow($"DebugImg({i})", partImages[i]);
                }
            }

            Console.WriteLine("---------- step: crnnNet getTextLines ----------");
            var textLines = _crnnNet.GetTextLines(partImages);
            //textLines.ForEach(x => Console.WriteLine(x));

            var textBlocks = new List<TextBlock>();
            for (int i = 0; i < textLines.Count; ++i)
            {
                var textBlock = new TextBlock
                {
                    BoxPoints = textBoxes[i].Points,
                    BoxScore = textBoxes[i].Score,
                    AngleIndex = angles[i].Index,
                    AngleScore = angles[i].Score,
                    AngleTime = angles[i].Time,
                    Text = textLines[i].Text,
                    CharScores = textLines[i].CharScores,
                    CrnnTime = textLines[i].Time,
                    BlockTime = angles[i].Time + textLines[i].Time
                };
                textBlocks.Add(textBlock);
            }
            //textBlocks.ForEach(x => Console.WriteLine(x));

            watcher.Stop();
            var fullDetectTime = watcher.ElapsedMilliseconds;
            //Console.WriteLine($"fullDetectTime({fullDetectTime}ms)");

            //cropped to original size
            //Mat rgbBoxImg = new Mat(textBoxPaddingImg, originRect);
            //Mat boxImg = new Mat();
            //CvInvoke.CvtColor(rgbBoxImg, boxImg, ColorConversion.Rgb2Bgr);//convert to BGR for Output Result Img
            //CvInvoke.Imshow("Result", boxImg);

            var strRes = new StringBuilder();
            textBlocks.ForEach(x => strRes.AppendLine(x.Text));

            var ocrResult = new OcrResult
            {
                TextBlocks = textBlocks,
                DbNetTime = dbNetTime,
                BoxImg = textBoxPaddingImg,
                DetectTime = fullDetectTime,
                StrRes = strRes.ToString()
            };

            return ocrResult;
        }
    }
}
