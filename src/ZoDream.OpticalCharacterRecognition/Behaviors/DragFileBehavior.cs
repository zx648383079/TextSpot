using Microsoft.Xaml.Behaviors;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;

namespace ZoDream.Shared.Behaviors
{
    public class DragFileBehavior : Behavior<FrameworkElement>
    {
        public ICommand Command {
            get { return (ICommand)GetValue(CommandProperty); }
            set { SetValue(CommandProperty, value); }
        }

        // Using a DependencyProperty as the backing store for Command.  This enables animation, styling, binding, etc...
        public static readonly DependencyProperty CommandProperty =
            DependencyProperty.Register("Command", typeof(ICommand), typeof(DragFileBehavior), new PropertyMetadata(null));




        protected override void OnAttached()
        {
            base.OnAttached();
            AssociatedObject.PreviewDragOver += AssociatedObject_PreviewDragOver;
            AssociatedObject.Drop += AssociatedObject_Drop;
        }

        private void AssociatedObject_Drop(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                return;
            }
            var items = (IEnumerable<string>)e.Data.GetData(DataFormats.FileDrop);
            if (items == null)
            {
                return;
            }
            Command?.Execute(items);
        }

        private void AssociatedObject_PreviewDragOver(object sender, DragEventArgs e)
        {
            e.Effects = DragDropEffects.Link;
            e.Handled = true;
        }

        protected override void OnDetaching()
        {
            base.OnDetaching();
            AssociatedObject.PreviewDragOver -= AssociatedObject_PreviewDragOver;
            AssociatedObject.Drop -= AssociatedObject_Drop;
        }
    }
}
