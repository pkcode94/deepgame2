#pragma once

#ifdef _WIN32
  #define EXPORT_API __declspec(dllexport)
#else
  #define EXPORT_API
#endif

#include <cstdint>

extern "C" {

// Opaque handles
typedef void* CoreHandle;
typedef void* OpponentHandle;
typedef void* PathGateHandle;
typedef void* AgentHandle;

// Core
EXPORT_API CoreHandle CreateCore(int hiddenSize, int maxDepth, int agentCount);
EXPORT_API void DestroyCore(CoreHandle h);
EXPORT_API int CoreGetHiddenSize(CoreHandle h);
// Forward: input is float array length inputLen, output buffer of length outputLen (caller allocates)
EXPORT_API void CoreForward(CoreHandle h, const float* input, int inputLen, float* output, int outputLen);

// Opponent
EXPORT_API OpponentHandle CreateOpponent(int hiddenSize, int maxDepth);
EXPORT_API void DestroyOpponent(OpponentHandle h);
EXPORT_API void OpponentForward(OpponentHandle h, const float* input, int inputLen, float* output, int outputLen);

// PathGate
EXPORT_API PathGateHandle CreatePathGate(int hiddenSize, int basisCount, int depth);
EXPORT_API void DestroyPathGate(PathGateHandle h);
EXPORT_API void PathGateForward(PathGateHandle h, const float* input, int inputLen, float* output, int outputLen);

// Agent
EXPORT_API AgentHandle CreateAgent(int hiddenSize, int maxDepth, int reasoningOrder);
EXPORT_API void DestroyAgent(AgentHandle h);
EXPORT_API void AgentChooseMove(AgentHandle h, const float* state, int stateLen, int* moveOut);

// Utility
EXPORT_API void SetSeed(int seed);

}
