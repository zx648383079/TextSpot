using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace ZoDream.OpticalCharacterRecognition.OcrNet
{
    internal class BitmapDataView: IDataView
    {
        private readonly IEnumerable<Bitmap> _bitmap;

        public BitmapDataView(params Bitmap[] bitmap)
        {
            _bitmap = bitmap;

            var builder = new DataViewSchema.Builder();
            builder.AddColumn("image", new ImageDataViewType());
            Schema = builder.ToSchema();
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema { get; }

        public long? GetRowCount() => null;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null) =>
            new Cursor(this);

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };

        private sealed class Cursor : DataViewRowCursor
        {
            private bool _disposed;
            private long _position;
            private readonly IEnumerator<Bitmap> _enumerator;
            private readonly Delegate[] _getters;

            public override long Position => _position;
            public override long Batch => 0;
            public override DataViewSchema Schema { get; }

            public Cursor(BitmapDataView parent)
            {
                Schema = parent.Schema;
                _position = -1;
                _enumerator = parent._bitmap.GetEnumerator();
                _getters = new Delegate[]
                {
                    (ValueGetter<Bitmap>)ImageGetterImplementation
                };
            }

            protected override void Dispose(bool disposing)
            {
                if (_disposed)
                    return;
                if (disposing)
                {
                    _enumerator.Dispose();
                    _position = -1;
                }
                _disposed = true;
                base.Dispose(disposing);
            }

            private void ImageGetterImplementation(ref Bitmap value)
                => value = _enumerator.Current;

            private void IdGetterImplementation(ref DataViewRowId id)
                => id = new DataViewRowId((ulong)_position, 0);

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
                if (_enumerator.MoveNext())
                {
                    _position++;
                    return true;
                }
                Dispose();
                return false;
            }
        }
    }
}
