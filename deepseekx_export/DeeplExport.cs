using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace deepseekx_export
{
    // Thin static wrapper to expose factory methods for types from the main project
    // or native implementations. Prefer native (C++) handles for GPU acceleration.
    public static class Exported
    {
        // Managed wrapper around native core handle
        public sealed class NativeCore : IDisposable
        {
            public IntPtr Handle { get; private set; }
            public NativeCore(int hiddenSize, int maxDepth, int agentCount)
            {
                Handle = Native.CreateCore(hiddenSize, maxDepth, agentCount);
            }
            public int HiddenSize => Native.CoreGetHiddenSize(Handle);
            public float[] Forward(float[] input, int outLen)
            {
                var outArr = new float[outLen];
                Native.CoreForward(Handle, input, input.Length, outArr, outArr.Length);
                return outArr;
            }
            public void Dispose() { if (Handle != IntPtr.Zero) { Native.DestroyCore(Handle); Handle = IntPtr.Zero; } }
        }

        public static NativeCore CreateNativeCore(int hiddenSize, int maxDepth, int agentCount)
        {
            return new NativeCore(hiddenSize, maxDepth, agentCount);
        }

        // Fallback managed factory (existing managed implementation)
        public static MultiAgentFractalCore CreateManagedCore(int hiddenSize, int maxDepth, int agentCount)
        {
            return new MultiAgentFractalCore(hiddenSize, maxDepth, agentCount);
        }

        // Expose other factories for convenience
        public static FractalOpponent CreateOpponent(int hiddenSize, int maxDepth)
        {
            return new FractalOpponent(hiddenSize, maxDepth);
        }

        public static CombinatorialPathGate CreatePathGate(int hiddenSize, int basisCount, int depth)
        {
            return new CombinatorialPathGate(hiddenSize, basisCount, depth);
        }

        public static FractalAgent CreateAgent(int hiddenSize, int maxDepth, int reasoningOrder)
        {
            return new FractalAgent(hiddenSize, maxDepth, reasoningOrder);
        }
    }
}
