public static class RegimeClassifier
{
    public enum Regime
    {
        Quiescent = 0,   // ~0.15–0.20 Jy
        Flaring = 1    // ~1.12 Jy
    }

    public static int Classify(double flux)
    {
        if (flux > 0.6) return (int)Regime.Flaring;
        return (int)Regime.Quiescent;
    }
}
