using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
using ZoDream.OpticalCharacterRecognition.ViewModels;

namespace ZoDream.OpticalCharacterRecognition.Pages
{
    /// <summary>
    /// OCRPage.xaml 的交互逻辑
    /// </summary>
    public partial class OCRPage : Page
    {
        public OCRPage()
        {
            InitializeComponent();
        }

        public OCRViewModel ViewModel => (OCRViewModel)DataContext;

        private void Page_Unloaded(object sender, RoutedEventArgs e)
        {
            ViewModel?.Dispose();
        }
    }
}
