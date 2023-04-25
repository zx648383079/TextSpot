using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using ZoDream.OpticalCharacterRecognition;

namespace ZoDream.Shared.Routes
{
    public class ShellService : IDisposable
    {
        private Frame? InnerFrame;
        private Action? AsyncNavigateFn;
        private readonly Dictionary<string, ShellRoute> Routes = new();

        private readonly Dictionary<string, FrameworkElement> Histories = new();
        private readonly List<string> HistoryRoutes = new();

        public FrameworkElement? Current { get; private set; }
        public ShellRoute? CurrentRoute { get; private set; }

        public void RegisterRoute(string routeName, Type page)
        {
            if (Routes.ContainsKey(routeName))
            {
                Routes[routeName] = new ShellRoute(routeName, page);
                Histories.Remove(routeName);
            } else
            {
                Routes.Add(routeName, new ShellRoute(routeName, page));
            }
        }

        public void RegisterRoute(string routeName, Type page, Type viewModel)
        {
            if (Routes.ContainsKey(routeName))
            {
                Routes[routeName] = new ShellRoute(routeName, page, viewModel);
                Histories.Remove(routeName);
            }
            else
            {
                Routes.Add(routeName, new ShellRoute(routeName, page, viewModel));
            }
        }

        public async Task GoBackAsync()
        {
            if (HistoryRoutes.Count < 2 || InnerFrame is null)
            {
                return;
            }
            var routeName = HistoryRoutes[HistoryRoutes.Count - 2];
            if (routeName == null)
            {
                return;
            }
            await GoToAsync(routeName);
        }

        public async Task GoToAsync(string routeName, IDictionary<string, object> queries)
        {
            if (InnerFrame is null)
            {
                AsyncNavigateFn = () => {
                    _ = GoToAsync(routeName, queries);
                };
                return;
            }
            await LoadAsync(routeName);
            if (Current is not null && Current.DataContext is IQueryAttributable o)
            {
                o.ApplyQueryAttributes(queries);
            }
        }

        public async Task GoToAsync(string routeName)
        {
            if (CurrentRoute is not null && CurrentRoute.Name == routeName)
            {
                return;
            }
            Current = null;
            if (InnerFrame is null)
            {
                AsyncNavigateFn = () => {
                    _ = LoadAsync(routeName);
                };
                return;
            }
            await LoadAsync(routeName);
            if (Current is not null && Current.DataContext is IQueryAttributable o)
            {
                o.ApplyQueryAttributes();
            }
        }

        private async Task LoadAsync(string routeName)
        {
            if (CurrentRoute is not null && CurrentRoute.Name == routeName)
            {
                return;
            }
            Current = null;
            if (InnerFrame is null)
            {
                return;
            }
            if (!Routes.TryGetValue(routeName, out var route))
            {
                return;
            }
            await App.Current.Dispatcher.InvokeAsync(() => {
                var page = CreatePage(route);
                if (!InnerFrame.Navigate(page))
                {
                    return;
                }
                var hIndex = HistoryRoutes.IndexOf(route.Name);
                if (hIndex < 0)
                {
                    HistoryRoutes.Add(route.Name);
                }
                else if (hIndex < HistoryRoutes.Count - 1)
                {
                    HistoryRoutes.RemoveRange(hIndex + 1, HistoryRoutes.Count - hIndex - 1);
                }
                if (page is FrameworkElement o)
                {
                    Current = o;
                    CurrentRoute = route;
                }
                if (route.DataContext is not null && Current is not null)
                {
                    if (Current.DataContext is null)
                    {
                        Current.DataContext = Activator.CreateInstance(route.DataContext);
                    }
                }
            });
        }

        private object? CreatePage(ShellRoute route)
        {
            if (Histories.TryGetValue(route.Name, out var page))
            {
                return page;
            }
            var instance = Activator.CreateInstance(route.Page);
            if (instance is FrameworkElement o)
            {
                Histories.Add(route.Name, o);
            }
            return instance;
        }

        public void Bind(Frame frame)
        {
            InnerFrame = frame;
            if (AsyncNavigateFn is not null)
            {
                AsyncNavigateFn.Invoke();
                AsyncNavigateFn = null;
            }
        }

        public void Dispose()
        {
            Routes.Clear();
            Histories.Clear();
        }
    }

}
