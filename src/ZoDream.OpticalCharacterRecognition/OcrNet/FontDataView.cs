using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ZoDream.OpticalCharacterRecognition.OcrNet
{
    internal class FontDataView: IDataView
    {
        private readonly FontDataLoader _loader;

        const int FontWidth = 64;
        const int FontHeight = 64;
        const int FontSize = 36;

        public FontDataView(FontDataLoader loader)
        {
            _loader = loader;

            var builder = new DataViewSchema.Builder();
            builder.AddColumn("Label", TextDataViewType.Instance);
            builder.AddColumn("Image", new ImageDataViewType(FontWidth, FontHeight));
            Schema = builder.ToSchema();
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema { get; }

        public long? GetRowCount() => _loader.Count;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null) =>
            new Cursor(this, columnsNeeded.Any(c => c.Index == 0),
                    columnsNeeded.Any(c => c.Index == 1));

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };

        private SKBitmap Generate(uint val)
        {
            var bitmap = new SKBitmap(FontWidth, FontHeight);
            var canvas = new SKCanvas(bitmap);
            canvas.Clear(SKColors.White);
            var text = FontDataLoader.ToChar(val);
            var textPaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = 1,
                FakeBoldText = true,
                Color = SKColors.Black
            };
            var textWidth = textPaint.MeasureText(text);
            textPaint.TextSize = 0.95f * FontSize * textPaint.TextSize / textWidth;

            // Find the text bounds
            var textBounds = new SKRect();
            textPaint.MeasureText(text, ref textBounds);
            var xText = FontWidth/ 2 - textBounds.MidX;
            var yText = FontHeight / 2 - textBounds.MidY;
            canvas.DrawText(text, xText, yText, textPaint);
            return bitmap;
        }

        private sealed class Cursor : DataViewRowCursor
        {
            private bool _disposed;
            private readonly FontDataView _parent;
            private readonly Delegate[] _getters;
            public override long Position => _parent._loader.Position;
            public override long Batch => 0;
            public override DataViewSchema Schema => _parent.Schema;

            public Cursor(FontDataView parent, bool wantLabel, bool wantImage)
            {
                _parent = parent;
                _getters = new Delegate[]
                {
                    wantLabel ? (ValueGetter<string>)LabelGetterImplementation : null,
                    wantImage ? (ValueGetter<SKBitmap>)ImageGetterImplementation : null,
                };
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _parent._loader.Reset();
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            private void LabelGetterImplementation(ref string value)
            {
                value = FontDataLoader.ToChar((uint)_parent._loader.Current);
            }
            private void ImageGetterImplementation(ref SKBitmap value)
            {
                value = _parent.Generate((uint)_parent._loader.Current);
            }

            private void IdGetterImplementation(ref DataViewRowId id)
                => id = new DataViewRowId((ulong)Position, 0);

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (!IsColumnActive(column))
                    throw new ArgumentOutOfRangeException(nameof(column));
                return (ValueGetter<TValue>)_getters[column.Index];
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => IdGetterImplementation;

            public override bool IsColumnActive(DataViewSchema.Column column)
                => _getters[column.Index] != null;


            public override bool MoveNext()
            {
                if (_disposed)
                    return false;
                if (_parent._loader.MoveNext())
                {
                    return true;
                }
                Dispose();
                return false;
            }
        }
    }
}
