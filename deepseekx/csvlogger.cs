using System;
using System.Globalization;
using System.IO;

public sealed class CsvLogger : IDisposable
{
    private readonly StreamWriter writer;

    public CsvLogger(string path)
    {
        writer = new StreamWriter(string.Join(@"C:\Users\patri\Desktop\deepseekx-master\deepseekx\bin\Debug\net10.0\win-x64\",path), append: false);
        writer.WriteLine("epoch,step,input,expected,predicted,regime");
        writer.Flush();
    }

    public void Log(
    int epoch,
    int step,
    double input,
    double expected,
    double predicted,
    int regime,
    int expert)
    {
        writer.WriteLine(
            string.Format(
                CultureInfo.InvariantCulture,
                "{0},{1},{2:F4},{3:F4},{4:F4},{5},{6}",
                epoch, step, input, expected, predicted, regime, expert
            )
        );
    }


    public void Dispose()
    {
        writer.Flush();
        writer.Dispose();
    }
}
