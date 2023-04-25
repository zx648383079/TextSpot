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
using ZoDream.Studio.Extensions;

namespace ZoDream.OpticalCharacterRecognition.Controls
{
    public class ImagePreview : Control
    {
        static ImagePreview()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(ImagePreview), new FrameworkPropertyMetadata(typeof(ImagePreview)));
        }

        public object? ImageSource
        {
            get { return GetValue(ImageSourceProperty); }
            set { SetValue(ImageSourceProperty, value); }
        }

        // Using a DependencyProperty as the backing store for ImageSource.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ImageSourceProperty =
            DependencyProperty.Register("ImageSource", typeof(object), typeof(ImagePreview), new PropertyMetadata(null, (d, s) =>
            {
                (d as ImagePreview)?.InvalidateVisual();
            }));



        public int ImageWidth
        {
            get { return (int)GetValue(ImageWidthProperty); }
            set { SetValue(ImageWidthProperty, value); }
        }

        // Using a DependencyProperty as the backing store for ImageWidth.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ImageWidthProperty =
            DependencyProperty.Register("ImageWidth", typeof(int), typeof(ImagePreview), new PropertyMetadata(0));



        public int ImageHeight
        {
            get { return (int)GetValue(ImageHeightProperty); }
            set { SetValue(ImageHeightProperty, value); }
        }

        // Using a DependencyProperty as the backing store for ImageHeight.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty ImageHeightProperty =
            DependencyProperty.Register("ImageHeight", typeof(int), typeof(ImagePreview), new PropertyMetadata(0));



        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            if (ActualWidth == 0 || ActualHeight == 0)
            {
                return;
            }
            var pen = new Pen(BorderBrush, BorderThickness.Top);
            drawingContext.DrawRectangle(Background, pen, new Rect(0, 0, ActualWidth, ActualHeight));
            UpdatePreview(drawingContext);
        }



        private void UpdatePreview(DrawingContext context)
        {
            var imageWidth = ImageWidth;
            var imageHeight = ImageHeight;
            if (imageWidth == 0 || imageWidth == 0)
            {
                if (ImageSource is System.Drawing.Bitmap bit)
                {
                    imageHeight = bit.Height;
                    imageWidth = bit.Width;
                }
                else if (ImageSource is BitmapSource b)
                {
                    imageHeight = b.PixelHeight;
                    imageWidth = b.PixelWidth;
                }
            }
            if (ActualWidth == 0 || imageHeight == 0 || imageWidth == 0)
            {
                return;
            }
            var width = ActualWidth;
            var height = ActualHeight;

            var (w, h) = GetSize(width, height, imageWidth, imageHeight);

            ImageSource? innerImage = null;
            if (ImageSource is System.Drawing.Bitmap o)
            {
                innerImage = o.ToBitmapSource();
            }
            else if (ImageSource is ImageSource j)
            {
                innerImage = j;
            }
            if (innerImage is null)
            {
                return;
            }
            context.DrawImage(innerImage, new Rect(
                (ActualWidth - w) / 2,
                (ActualHeight - h) / 2,
                w, h
                ));
        }

        private (double, double) GetSize(double maxWidth, double maxHeight, double imageWidth, double imageHeight)
        {
            var wScale = maxWidth / imageWidth;
            var hScale = maxHeight / imageHeight;
            if (wScale < hScale)
            {
                return (maxWidth, imageHeight * wScale);
            }
            else
            {
                return (imageWidth * hScale, maxHeight);
            }
        }
    }
}
