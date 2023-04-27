using Microsoft.ML.Data;

namespace ZoDream.OpticalCharacterRecognition.OcrNet.OnnxPrediction.DataModel
{
    public class TinyYoloPrediction : IOnnxObjectPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels { get; set; }
    }
}
