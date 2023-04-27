using Emgu.CV;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using ZoDream.OpticalCharacterRecognition.OcrNet.DataModels;
using ZoDream.OpticalCharacterRecognition.OcrNet.OnnxPrediction;
using ZoDream.OpticalCharacterRecognition.OcrNet.OnnxPrediction.DataModel;

namespace ZoDream.OpticalCharacterRecognition.OcrNet
{
    public class OCRContext
    {
        private readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "data", "font_model.zip");
        public readonly string FontInputPath = Path.Combine(Environment.CurrentDirectory, "inputs");

        private readonly MLContext Context = new();

        public void TrainAndSave(FontDataLoader loader)
        {
            var trainDataView = new FontDataView(loader);
            var pipelineEstimator = Context.MulticlassClassification.Trainers.ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "Label"
                                         //,validationSet: trainDataView
                                         )
                                        .Append(Context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));
            var trainedModel = pipelineEstimator.Fit(trainDataView);
            Context.Model.Save(trainedModel, trainDataView.Schema, ModelPath);
            //using var fs = File.Create(ONNXModelPath);
            //Context.Model.ConvertToOnnx(trainedModel, trainData, fs);
        }
        public IList<YoloBoundingBox> Prediction(string fileName)
        {
            return Prediction((Bitmap)Image.FromFile(fileName));
        }

        public IList<YoloBoundingBox> Prediction(Bitmap src)
        {
            var customVisionModel = new CustomVisionModel(ModelPath);
            var modelConfigurator = new OnnxModelConfigurator(customVisionModel);

            var outputParser = new OnnxOutputParser(customVisionModel);
            var customVisionPredictionEngine = modelConfigurator.GetMlNetPredictionEngine<CustomVisionPrediction>();
            var frame = new ImageInputData { Image = src };
            var labels = customVisionPredictionEngine?.Predict(frame).PredictedLabels;
            var boundingBoxes = outputParser.ParseOutputs(labels);
            var filteredBoxes = outputParser.FilterBoundingBoxes(boundingBoxes, 5, 0.5f);
            // DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);
            return filteredBoxes;
        }

        public IList<YoloBoundingBox> Prediction(Mat src)
        {
            throw new NotImplementedException();
            //var model = Context.Model.Load(ModelPath, out var inputSchema);
            //var predictionEngine = Context.Model.CreatePredictionEngine<CheckData, CheckPrediction>(model);

            //var test = new CheckData
            //{

            //};
            //var prediction = predictionEngine.Predict(test);
        }

        private ITransformer BuildAndTrainModel(IDataView splitTrainSet)
        {
            var estimator = Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(FontImageData.Label))
                .Append(Context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            // 开始生成模型
            var model = estimator.Fit(splitTrainSet);
            // 生成完成
            Context.Model.Save(model, splitTrainSet.Schema, ModelPath);
            return model;
        }

        private void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            var image = (Bitmap)Image.FromFile(Path.Combine(inputImageLocation, imageName));
            DrawBoundingBox(image, filteredBoundingBoxes);
            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }
            image.Save(Path.Combine(outputImageLocation, imageName));

        }
        private void DrawBoundingBox(Bitmap src, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            var originalImageHeight = src.Height;
            var originalImageWidth = src.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                // Get Bounding Box Dimensions
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                // Resize To Image
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

                // Bounding Box Text
                var text = box.Description;

                using var thumbnailGraphic = Graphics.FromImage(src);
                thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // Define Text Options
                var drawFont = new Font("Arial", 12, FontStyle.Bold);
                var size = thumbnailGraphic.MeasureString(text, drawFont);
                var fontBrush = new SolidBrush(Color.Black);
                var atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                // Define BoundingBox options
                var pen = new Pen(box.BoxColor, 3.2f);
                var colorBrush = new SolidBrush(box.BoxColor);

                // Draw text on image 
                thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                // Draw bounding box on image
                thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
            }

            
        }

    }
}
