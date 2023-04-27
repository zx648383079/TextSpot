using Microsoft.ML.Data;

namespace ZoDream.OpticalCharacterRecognition.OcrNet.OnnxPrediction.DataModel
{
    public class CustomVisionPrediction : IOnnxObjectPrediction
    {
        [ColumnName("model_outputs0")]
        public float[] PredictedLabels { get; set; }
    }
}
