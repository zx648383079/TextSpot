using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Tesseract;

namespace TextSpot
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private CancellationTokenSource messageToken = new CancellationTokenSource();
        private TesseractEngine tesseractEngine;

        private void ClearBtn_Click(object sender, RoutedEventArgs e)
        {
            TextTb.Text = "";
        }

        private void SpotBtn_Click(object sender, RoutedEventArgs e)
        {
            Hide();
            var page = new SpotWindow();
            page.Spot += (_, image) =>
            {
                page.Close();
                Show();
                Render(image);
                image.Dispose();
                TextTb.Focus();
            };
            page.Show();
            page.Activate();
        }

        private void CopyBtn_Click(object sender, RoutedEventArgs e)
        {
            Clipboard.SetData(DataFormats.Text, TextTb.Text); //复制内容到剪切
        }

        private void Toast(string message)
        {
            MessageTb.Text = message;
            messageToken.Cancel();
            _ = Task.Factory.StartNew(() =>
              {
                  Thread.Sleep(2000);
                  Application.Current.Dispatcher.Invoke(() =>
                  {
                      MessageTb.Text = "";
                  });
              }, messageToken.Token);
        }

        private void Window_Unloaded(object sender, RoutedEventArgs e)
        {
            tesseractEngine.Dispose();
        }

        private void initEngine()
        {
            if (tesseractEngine != null)
            {
                return;
            }
            tesseractEngine = new TesseractEngine(@"./tessdata", "chi_sim+eng", EngineMode.Default);

        }

        private void Render(System.Drawing.Bitmap bitmap)
        {
            if (bitmap == null)
            {
                Toast("识别识别！");
                return;
            }
            initEngine();
            using (var img = PixConverter.ToPix(bitmap))
            {
                using (var page = tesseractEngine.Process(img))
                {
                    var text = page.GetText();
                    TextTb.Text = text;
                    Toast("识别成功！");
                }
            }
        }

        private void Render(string file)
        {
            initEngine();
            using (var img = Pix.LoadFromFile(file))
            {
                using (var page = tesseractEngine.Process(img))
                {
                    var text = page.GetText();
                    TextTb.Text = text;
                    Toast("识别成功！");
                }
            }
        }

        private void OpenBtn_Click(object sender, RoutedEventArgs e)
        {
            var open = new Microsoft.Win32.OpenFileDialog
            {
                Multiselect = true,
                Filter = "图片|*.png;*.jpg|所有文件|*.*",
                Title = "选择图片"
            };
            if (open.ShowDialog() == true)
            {
                Render(open.FileName);
            }
        }

        private void TextTb_Drop(object sender, DragEventArgs e)
        {
            var fileName = ((System.Array)e.Data.GetData(DataFormats.FileDrop)).GetValue(0).ToString();
            var ext = System.IO.Path.GetExtension(fileName); 
            if (ext != ".png" && ext != "jpg")
            {
                return;
            }
            Render(fileName);
        }

        private void TextTb_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Link;
            } else
            {
                e.Effects = DragDropEffects.None;
            }
        }
    }
}