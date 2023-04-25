using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;

namespace ZoDream.OpticalCharacterRecognition.OcrLite
{
    public class AngleNet
    {
        private readonly float[] MeanValues = { 127.5F, 127.5F, 127.5F };
        private readonly float[] NormValues = { 1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F };
        private const int _angleDstWidth = 192;
        private const int _angleDstHeight = 32;
        private const int _angleCols = 2;
        private InferenceSession? _angleNet;

        public AngleNet() { }

        ~AngleNet()
        {
            _angleNet?.Dispose();
        }

        public void InitModel(string path, int numThread)
        {
            try
            {
                var op = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    InterOpNumThreads = numThread,
                    IntraOpNumThreads = numThread
                };
                _angleNet = new InferenceSession(path);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public List<Angle> GetAngles(List<Bitmap> partImgs, bool doAngle, bool mostAngle)
        {
            var angles = new List<Angle>();
            if (doAngle)
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var startTicks = DateTime.Now.Ticks;
                    var angle = GetAngle(partImgs[i]);
                    var endTicks = DateTime.Now.Ticks;
                    var angleTime = (endTicks - startTicks) / 10000F;
                    angle.Time = angleTime;
                    angles.Add(angle);
                }
            }
            else
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var angle = new Angle
                    {
                        Index = -1,
                        Score = 0F
                    };
                    angles.Add(angle);
                }
            }
            //Most Possible AngleIndex
            if (doAngle && mostAngle)
            {
                var angleIndexes = new List<int>();
                angles.ForEach(x => angleIndexes.Add(x.Index));

                double sum = angleIndexes.Sum();
                double halfPercent = angles.Count / 2.0f;
                int mostAngleIndex;
                if (sum < halfPercent)
                {//all angle set to 0
                    mostAngleIndex = 0;
                }
                else
                {//all angle set to 1
                    mostAngleIndex = 1;
                }
                Console.WriteLine($"Set All Angle to mostAngleIndex({mostAngleIndex})");
                for (int i = 0; i < angles.Count; ++i)
                {
                    Angle angle = angles[i];
                    angle.Index = mostAngleIndex;
                    angles[i] = angle;
                }
            }
            return angles;
        }

        private Angle GetAngle(Bitmap src)
        {
            var angle = new Angle();
            var angleImg = AdjustTargetImg(src, _angleDstWidth, _angleDstHeight);
            var inputTensors = OcrUtils.SubstractMeanNormalize(angleImg, MeanValues, NormValues);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensors)
            };
            try
            {
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _angleNet.Run(inputs);
                var resultsArray = results.ToArray();
                Console.WriteLine(resultsArray);
                float[] outputData = resultsArray[0].AsEnumerable<float>().ToArray();
                return ScoreToAngle(outputData, _angleCols);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                //throw ex;
            }
            return angle;

        }

        private Angle ScoreToAngle(float[] srcData, int angleCols)
        {
            var angle = new Angle();
            int angleIndex = 0;
            float maxValue = -1000.0F;
            for (int i = 0; i < angleCols; i++)
            {
                if (i == 0) maxValue = srcData[i];
                else if (srcData[i] > maxValue)
                {
                    angleIndex = i;
                    maxValue = srcData[i];
                }
            }
            angle.Index = angleIndex;
            angle.Score = maxValue;
            return angle;
        }

        /// <summary>
        /// 以高为准缩放比例，右边填补白色
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dstWidth"></param>
        /// <param name="dstHeight"></param>
        /// <returns></returns>
        private Bitmap AdjustTargetImg(Bitmap src, int dstWidth, int dstHeight)
        {
            var srcFit = new Bitmap(dstHeight, dstWidth, PixelFormat.Format24bppRgb);
            var scale = (float)src.Height / dstHeight;
            using var g = Graphics.FromImage(srcFit);
            g.Clear(Color.White);
            g.DrawImage(src, new Rectangle(
                0, 0, dstWidth, dstHeight
                ), new Rectangle(0,0, (int)Math.Min(src.Width / scale, dstWidth * scale), src.Height), GraphicsUnit.Pixel);
            g.Save();
            return srcFit;
        }
    }
}
