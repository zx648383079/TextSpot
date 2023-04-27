using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Input;
using ZoDream.Shared.Storage;
using ZoDream.Shared.ViewModel;
using ZoDream.Shared.ViewModels;

namespace ZoDream.OpticalCharacterRecognition.ViewModels
{
    public class OCRViewModel: BindableBase, IDisposable
    {
        public OCRViewModel()
        {
            DragCommand = new RelayCommand(TapOpenDrag);
            SaveAsCommand = new RelayCommand(TapSaveAs);
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

        private bool isAppend;

        public bool IsAppend {
            get => isAppend;
            set => Set(ref isAppend, value);
        }

        private string text = string.Empty;

        public string Text {
            get => text;
            set => Set(ref text, value);
        }
        public ICommand DragCommand { get; private set; }
        public ICommand SaveAsCommand { get; private set; }

        private void TapSaveAs(object? _)
        {
            var picker = new Microsoft.Win32.SaveFileDialog
            {
                RestoreDirectory = true,
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                Filter = "文本文件|*.txt|所有文件|*.*"
            };
            if (picker.ShowDialog() != true)
            {
                return;
            }
            _ = LocationStorage.WriteAsync(picker.FileName, Text);
        }
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
            Text = IsAppend && !string.IsNullOrWhiteSpace(Text) ? $"{Text}\n{res.StrRes}" : res.StrRes;
            ImageBitmap = res.BoxImg.ToBitmap();
        }

        public void Dispose()
        {
            ImageBitmap?.Dispose();
        }
    }
}
