using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace ZoDream.Studio.Extensions
{
    public static class ListExtension
    {
        public static ObservableCollection<T> ToCollection<T>(this IEnumerable<T> items)
        {
            var data = new ObservableCollection<T>();
            foreach (var item in items)
            {
                data.Add(item);
            }
            return data;
        }

        public static void ToCollection<T>(this IEnumerable<T> items, ICollection<T> parent)
        {
            parent.Clear();
            foreach (var item in items)
            {
                parent.Add(item);
            }
        }
    }
}
