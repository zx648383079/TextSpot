using System.Text;

namespace ZoDream.Tests
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            Assert.AreEqual(ToChar(39739), "еч");
        }

        public static string ToChar(ushort val)
        {
            var buffer = val <= short.MaxValue ? BitConverter.GetBytes((short)val) : BitConverter.GetBytes(val);
            return Encoding.Unicode.GetString(buffer);
        }
    }
}