using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace ZoDream.Shared.Routes
{
    public static class ShellManager
    {

        private static readonly ShellService Service = new();

        public static void RegisterRoute(string routeName, Type page)
        {
            Service.RegisterRoute(routeName, page);
        }

        public static void RegisterRoute(string routeName, Type page, Type viewModel)
        {
            Service.RegisterRoute(routeName, page, viewModel);
        }

        public static async void GoToAsync(string routeName, IDictionary<string, object> queries)
        {
            await Service.GoToAsync(routeName, queries);
        }

        public static async void GoToAsync(string routeName)
        {
            await Service.GoToAsync(routeName);
        }

        public static async void GoBackAsync()
        {
            await Service.GoBackAsync();
        }

        public static void BindFrame(Frame bodyPanel)
        {
            Service.Bind(bodyPanel);
        }

        public static void UnBind()
        {
            Service.Dispose();
        }
    }
}
