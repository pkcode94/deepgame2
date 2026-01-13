using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Linq;

public class EHTIntensityWrapper
{
    private readonly WordTokenizer _tokenizer;

    public EHTIntensityWrapper(WordTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    // Logic: Translate intensity into tokens based on the Photon Ring structure
    public string IntensityToToken(double flux)
    {
        // Based on normalized M87 intensity (0.0 to 1.0)
        if (flux > 0.8) return "peak";   // Main Ring
        if (flux > 0.5) return "high";   // Primary Sub-ring
        if (flux > 0.2) return "mid";    // Outer accretion
        return "low";                   // Shadow/Background
    }

    public List<double> LoadEHTCsv(string filePath)
    {
        var intensities = new List<double>();
        var lines = File.ReadAllLines(filePath);

        foreach (var line in lines.Skip(1)) // Skip header
        {
            var parts = line.Split(',');
            // Usually: Radius, Intensity
            if (parts.Length >= 2 && double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out double val))
            {
                intensities.Add(val);
            }
        }
        return intensities;
    }

    // This creates a "Fractal Prompt" from the black hole image data
    public string GenerateFractalPrompt(List<double> intensities)
    {
        // We look for local maxima to detect rings
        // Every time intensity goes UP, we open a bracket [ (entering a ring)
        // Every time it goes DOWN, we close a bracket ] (leaving a ring)
        string prompt = "horizon ";
        bool inRing = false;

        foreach (var val in intensities.Take(20)) // Take a sample profile
        {
            string token = IntensityToToken(val);
            if (token == "peak" && !inRing) { prompt += "[ "; inRing = true; }
            prompt += token + " ";
            if (token == "low" && inRing) { prompt += "] "; inRing = false; }
        }
        return prompt.Trim();
    }
}