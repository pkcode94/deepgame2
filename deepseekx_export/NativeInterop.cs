using System;
using System.Runtime.InteropServices;

namespace deepseekx_export
{
    internal static class Native
n    {
        const string DLL = "deepseekx_native";

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreateCore(int hiddenSize, int maxDepth, int agentCount);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void DestroyCore(IntPtr h);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreGetHiddenSize(IntPtr h);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CoreForward(IntPtr h, float[] input, int inputLen, float[] output, int outputLen);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreateOpponent(int hiddenSize, int maxDepth);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void DestroyOpponent(IntPtr h);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void OpponentForward(IntPtr h, float[] input, int inputLen, float[] output, int outputLen);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreatePathGate(int hiddenSize, int basisCount, int depth);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void DestroyPathGate(IntPtr h);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PathGateForward(IntPtr h, float[] input, int inputLen, float[] output, int outputLen);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CreateAgent(int hiddenSize, int maxDepth, int reasoningOrder);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void DestroyAgent(IntPtr h);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void AgentChooseMove(IntPtr h, float[] state, int stateLen, int[] moveOut);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void SetSeed(int seed);
    }
}
