using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using static Program;

namespace deepseekx
{
    public class FractalAgent
    {

        private readonly MultiAgentFractalCore core;
        private readonly BoardEncoder encoder;
        private readonly int hiddenSize;
        private readonly int reasoningOrder;

        public FractalAgent(int hiddenSize, int maxDepth, int reasoningOrder = 1)
        {
            this.hiddenSize = hiddenSize;
            this.reasoningOrder = reasoningOrder;

            core = new MultiAgentFractalCore(hiddenSize, maxDepth, agentCount: 5);
            encoder = new BoardEncoder(hiddenSize);
        }

        public int ChooseMove(TicTacToe game)
        {
            var moves = game.GetLegalMoves();

            float bestScore = float.NegativeInfinity;
            int bestMove = moves[0];

            foreach (var move in moves)
            {
                var next = (Cell[])game.Board.Clone();
                next[move] = game.CurrentPlayer;

                var state = encoder.forward(next); // [1,H]

                // Ordnung der Inferenz wählbar
                var global = core.forward(state, orderK: reasoningOrder); // [1,H]
                var score = global.mean().ToSingle();

                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                }
            }

            return bestMove;
        }
    }
}
