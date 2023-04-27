using System;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ZoDream.OpticalCharacterRecognition.Controls;
using ZoDream.OpticalCharacterRecognition.OcrNet;
using ZoDream.Shared.ViewModel;
using ZoDream.Shared.ViewModels;

namespace ZoDream.OpticalCharacterRecognition.ViewModels
{
    public class TrainViewModel: BindableBase, IDisposable
    {
        public TrainViewModel()
        {
            ExecuteCommand = new RelayCommand(TapExecute);
            InkMinusCommand = new RelayCommand(TapInkMinus);
            InkPlusCommand = new RelayCommand(TapInkPlus);
            InkRefreshCommand = new RelayCommand(TapInkRefresh);
            PredictionCommand = new RelayCommand(TapPrediction);
        }

        public InkCanvas? InkPanel { get; set; }

        private readonly OCRContext Context = new();

        private string fontFileName = "微软雅黑";

        public string FontFileName {
            get => fontFileName;
            set => Set(ref fontFileName, value);
        }

        private string beginChar = string.Empty;

        public string BeginChar {
            get => beginChar;
            set => Set(ref beginChar, value);
        }

        private string endChar = string.Empty;

        public string EndChar {
            get => endChar;
            set => Set(ref endChar, value);
        }

        private Bitmap? imageBitmap;

        public Bitmap? ImageBitmap {
            get => imageBitmap;
            set => Set(ref imageBitmap, value);
        }

        private bool executeEnabled = true;

        public bool ExecuteEnabled {
            get => executeEnabled;
            set => Set(ref executeEnabled, value);
        }

        public ICommand ExecuteCommand { get; private set; }
        public ICommand InkMinusCommand { get; private set; }
        public ICommand InkPlusCommand { get; private set; }
        public ICommand InkRefreshCommand { get; private set; }

        public ICommand PredictionCommand { get; set; }

        private void TapPrediction(object? _)
        {

        }

        private void TapInkMinus(object? _)
        {
            if (InkPanel is null)
            {
                return;
            }
            if (InkPanel.DefaultDrawingAttributes.Width < 2)
            {
                return;
            }
            InkPanel.DefaultDrawingAttributes.Width--;
            InkPanel.DefaultDrawingAttributes.Height = InkPanel.DefaultDrawingAttributes.Width;
        }

        private void TapInkPlus(object? _)
        {
            if (InkPanel is null)
            {
                return;
            }
            if (InkPanel.DefaultDrawingAttributes.Width > 30)
            {
                return;
            }
            InkPanel.DefaultDrawingAttributes.Width++;
            InkPanel.DefaultDrawingAttributes.Height = InkPanel.DefaultDrawingAttributes.Width;
        }

        private void TapInkRefresh(object? _)
        {
            if (InkPanel is null)
            {
                return;
            }
            InkPanel.Strokes.Clear();
            InkPanel.Children.Clear();
        }

        private void TapExecute(object? _)
        {
            if (string.IsNullOrWhiteSpace(BeginChar))
            {
                return;
            }
            var loader = new FontDataLoader()
            {
                Font = FontFileName,
                Begin = FontDataLoader.ToInt(BeginChar),
            };
            if (!string.IsNullOrWhiteSpace(EndChar))
            {
                loader.End = FontDataLoader.ToInt(EndChar);
            }
            ExecuteEnabled = false;
            _ = GenerateFontAsync(loader);
        }

        private async Task GenerateFontAsync(FontDataLoader loader)
        {
            if (loader.Font.EndsWith(".ttf") && File.Exists(loader.Font))
            {
                loader.Font = await Utils.Font.GetFontFileAsync(loader.Font);
            }
            await Task.Factory.StartNew(() => {
                FontDataGenerator.Generate(loader, Context.FontInputPath);
                App.Current.Dispatcher.Invoke(() => {
                    ExecuteEnabled = true;
                });
            });
        }

        private void AddImage(Bitmap src)
        {
            if (InkPanel is null)
            {
                return;
            }
            InkPanel.Strokes.Clear();
            InkPanel.Children.Clear();
            InkPanel.Children.Add(new ImagePreview()
            {
                ImageSource = src,
            });
        }

        private Bitmap GetInkImage()
        {
            var strokes = InkPanel!.Strokes;
            var rect = strokes.GetBounds();
            var rtb = new RenderTargetBitmap((int)rect.Right,
                (int)rect.Bottom, 96, 96, System.Windows.Media.PixelFormats.Rgb24);
            rtb.Render(InkPanel);
            using var outStream = new MemoryStream();
            var enc = new BmpBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(rtb));
            enc.Save(outStream);
            return new Bitmap(outStream);
        }

        

        public void Dispose()
        {
            ImageBitmap?.Dispose();
            InkPanel = null;
        }
    }
}
