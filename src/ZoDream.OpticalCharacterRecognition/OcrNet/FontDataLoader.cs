using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace ZoDream.OpticalCharacterRecognition.OcrNet
{
    public class FontDataLoader: IEnumerable<uint>, IEnumerator<uint>
    {
        public string Font { get; set; } = string.Empty;

        public uint Begin { get; set; }

        public uint? End { get; set; }

        public int Position { get; private set; } = -1;

        public object Current => Position < 0 ? 0 : (uint)(Begin + Position);

        uint IEnumerator<uint>.Current => (uint)Current;

        public int Count => End is null ? 1 : (int)(End - Begin + 1);

        public uint? this[int index] => index < 0 || index >= Count ? null : (uint)(Begin + index);

        public void Dispose()
        {
            
        }

        public bool MoveNext()
        {
            if (Position >= Count - 1)
            {
                return false;
            }
            Position++;
            return true;
        }

        public void Reset()
        {
            Position = -1;
        }

        public IEnumerator<uint> GetEnumerator()
        {
            return this;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this;
        }


        public static string ToChar(uint val)
        {
            var buffer = val <= short.MaxValue ? BitConverter.GetBytes((short)val) : BitConverter.GetBytes(val);
            return Encoding.Unicode.GetString(buffer);
        }

        public static uint ToInt(string val)
        {
            val = val.Trim();
            if (val.StartsWith("\\u"))
            {
                val = Regex.Unescape(val);
            }
            var buffer = Encoding.Unicode.GetBytes(val);
            if (buffer.Length < 4)
            {
                return BitConverter.ToUInt16(buffer);
            }
            return BitConverter.ToUInt32(buffer, 0);
        }
    }
}
