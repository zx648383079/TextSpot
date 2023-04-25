using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using ZoDream.OpticalCharacterRecognition.OcrLite;
using ZoDream.Shared.ViewModel;
using ZoDream.Shared.ViewModels;

namespace ZoDream.OpticalCharacterRecognition.ViewModels
{
    public class OCRViewModel: BindableBase, IDisposable
    {
        public OCRViewModel()
        {
            DragCommand = new RelayCommand(TapOpenDrag);
            var baseFolder = AppDomain.CurrentDomain.BaseDirectory;
            Engine.InitModels(
                Path.Combine(baseFolder, "models", "dbnet.onnx"),
                Path.Combine(baseFolder, "models", "angle_net.onnx"),
                Path.Combine(baseFolder, "models", "crnn_lite_lstm.onnx"),
                Path.Combine(baseFolder, "models", "keys.txt"),
                4
                );
        }


        private readonly OcrLite.OcrLite Engine = new();

        private Bitmap? imageBitmap;

        public Bitmap? ImageBitmap {
            get => imageBitmap;
            set => Set(ref imageBitmap, value);
        }

        private string text = string.Empty;

        public string Text {
            get => text;
            set => Set(ref text, value);
        }
        public ICommand DragCommand { get; private set; }
        private void TapOpenDrag(object? arg)
        {
            if (arg is IEnumerable<string> items)
            {
                foreach (var item in items)
                {
                    LoadFile(item);
                    return;
                }
            }
        }


        private void LoadFile(string fileName)
        {
            try
            {
                ImageBitmap = (Bitmap)Image.FromFile(fileName);
            }
            catch (Exception)
            {
            }
            var res = Engine.Detect(fileName, 50, 1024, 0.618f, 0.3f, 2.0f, true, true);
            Text = res.StrRes;
            ImageBitmap = res.BoxImg.ToBitmap();
        }

        public void Dispose()
        {
            ImageBitmap?.Dispose();
        }
    }
}
