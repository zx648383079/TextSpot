using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Drawing;
using ZoDream.Shared.Storage;

namespace ZoDream.OpticalCharacterRecognition.OcrLite
{
    public class CrnnNet
    {
        private readonly float[] MeanValues = { 127.5F, 127.5F, 127.5F };
        private readonly float[] NormValues = { 1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F };
        private const int _crnnDstHeight = 32;
        private const int _crnnCols = 5531;

        private InferenceSession? _crnnNet;
        private List<string> _keys = new();

        public CrnnNet() { }

        ~CrnnNet()
        {
            _crnnNet?.Dispose();
        }

        public void InitModel(string path, string keysPath, int numThread)
        {
            try
            {
                var op = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    InterOpNumThreads = numThread,
                    IntraOpNumThreads = numThread
                };
                _crnnNet = new InferenceSession(path, op);
                _keys = InitKeys(keysPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }
        private List<string> InitKeys(string path)
        {
            var sr = LocationStorage.Reader(path);
            var keys = new List<string>();
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                //Console.WriteLine(line.ToString());
                keys.Add(line);
            }
            Console.WriteLine($"keys Size = {keys.Count}");
            return keys;
        }

        public List<TextLine> GetTextLines(List<Bitmap> partImgs)
        {
            var textLines = new List<TextLine>();
            for (int i = 0; i < partImgs.Count; i++)
            {
                var startTicks = DateTime.Now.Ticks;
                var textLine = GetTextLine(partImgs[i]);
                var endTicks = DateTime.Now.Ticks;
                var crnnTime = (endTicks - startTicks) / 10000F;
                textLine.Time = crnnTime;
                textLines.Add(textLine);
            }
            return textLines;
        }

        private TextLine GetTextLine(Bitmap src)
        {
            var textLine = new TextLine();

            var scale = (float)_crnnDstHeight / src.Height;
            int dstWidth = (int)(src.Width * scale);

            var srcResize = OcrUtils.Resize(src, dstWidth, _crnnDstHeight);
            var inputTensors = OcrUtils.SubstractMeanNormalize(srcResize, MeanValues, NormValues);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensors)
            };
            try
            {
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _crnnNet!.Run(inputs);
                var resultsArray = results.ToArray();
                Console.WriteLine(resultsArray);
                var outputData = resultsArray[0].AsEnumerable<float>().ToArray();

                var crnnRows = outputData.Count() / _crnnCols;

                return ScoreToTextLine(outputData, crnnRows, _crnnCols);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                //throw ex;
            }

            return textLine;
        }

        private TextLine ScoreToTextLine(float[] srcData, int rows, int cols)
        {
            var sb = new StringBuilder();
            var textLine = new TextLine();

            int lastIndex = 0;
            var scores = new List<float>();

            for (int i = 0; i < rows; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000F;

                //do softmax
                var expList = new List<float>();
                for (int j = 0; j < cols; j++)
                {
                    float expSingle = (float)Math.Exp(srcData[i * cols + j]);
                    expList.Add(expSingle);
                }
                float partition = expList.Sum();
                for (int j = 0; j < cols; j++)
                {
                    float softmax = expList[j] / partition;
                    if (softmax > maxValue)
                    {
                        maxValue = softmax;
                        maxIndex = j;
                    }
                }

                //no softmax
                /*for (int j = 0; j < cols; j++) {
                    if (srcData[i * cols + j] > maxValue) {
                        maxValue = srcData[i * cols + j];
                        maxIndex = j;
                    }
                }*/

                if (maxIndex > 0 && maxIndex < _keys.Count && (!(i > 0 && maxIndex == lastIndex)))
                {
                    scores.Add(maxValue);
                    sb.Append(_keys[maxIndex - 1]);
                }
                lastIndex = maxIndex;
            }
            textLine.Text = sb.ToString();
            textLine.CharScores = scores;
            return textLine;
        }

    }
}
