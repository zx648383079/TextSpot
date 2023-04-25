using Microsoft.Xaml.Behaviors;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace ZoDream.Shared.Behaviors
{
    public class ListItemDoubleClickBehavior: Behavior<ItemsControl>
    {
        #region Overrides

        protected override void OnAttached()
        {
            base.OnAttached();

            AssociatedObject.MouseDoubleClick += AssociatedObject_MouseDoubleClick;
        }

        protected override void OnDetaching()
        {
            base.OnDetaching();

            AssociatedObject.MouseDoubleClick -= AssociatedObject_MouseDoubleClick;
        }

        #endregion

        #region Private Methods

        private void AssociatedObject_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is not ItemsControl listView || e.OriginalSource is not DependencyObject originalSender)
            {
                return;
            }

            var container = ItemsControl.ContainerFromElement
                (sender as ItemsControl, e.OriginalSource as DependencyObject);

            if (container == null ||
                container == DependencyProperty.UnsetValue)
            {
                return;
            }

            var activatedItem = listView.ItemContainerGenerator.ItemFromContainer(container);

            if (activatedItem != null)
            {
                Invoke(activatedItem, e);
            }
        }



        #endregion

        #region Protected Methods

        protected void Invoke(object clickedItem, MouseButtonEventArgs parameter)
        {
            Command?.Execute(clickedItem);
        }

        #endregion

        #region Dependency Properties

        public static readonly DependencyProperty CommandProperty = DependencyProperty.Register(
            "Command", typeof(ICommand), typeof(ListItemDoubleClickBehavior),
            new PropertyMetadata(null));

        public ICommand Command {
            get { return (ICommand)GetValue(CommandProperty); }
            set { SetValue(CommandProperty, value); }
        }

        #endregion
    }
}
