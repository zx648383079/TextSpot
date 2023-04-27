using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace ZoDream.OpticalCharacterRecognition.Utils
{
    public static class Font
    {
        public static IList<string> GetSystemFont()
        {
            return Fonts.SystemFontFamilies.Select(i => i.Source).ToList();
        }

        public static async Task<string> GetFontFileAsync(string fileName)
        {
            var font = await FontInfo.Font.CreateAsync(fileName);
            return $"{fileName}#{font.Details.FullName}";
        }
    }
}
