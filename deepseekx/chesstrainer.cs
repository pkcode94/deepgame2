using System;
using System.Collections.Generic;
using System.Linq;
using ChessDotNet;
using TorchSharp;
using static TorchSharp.torch;

// Simple chess self-play trainer using MultiAgentFractalCore (two internal FractalOpponents).
// Runs self-play games between the two internal agents and performs a lightweight training step
// that rewards moves taken by the eventual winner.
public static class ChessTrainer
{
    // Encode a ChessGame into a tensor of shape [1, hiddenSize].
    // Simple encoding: for each square add piece value into vector slots (mod hiddenSize).
    private static Tensor EncodeBoard(ChessGame game, int hiddenSize)
    {
        var vec = new float[hiddenSize];

        // Use FEN parsing to avoid depending on Piece API differences
        string fen = game.GetFen();
        if (string.IsNullOrEmpty(fen)) return torch.zeros(1, hiddenSize);

        var parts = fen.Split(' ');
        var boardPart = parts.Length > 0 ? parts[0] : fen;

        int sq = 0; // 0..63 from a8 to h1 in FEN
        foreach (char ch in boardPart)
        {
            if (ch == '/') continue;
            if (char.IsDigit(ch))
            {
                int skip = ch - '0';
                sq += skip;
                continue;
            }

            float val = 0f;
            switch (char.ToLowerInvariant(ch))
            {
                case 'p': val = 1f; break;
                case 'n': val = 3f; break;
                case 'b': val = 3f; break;
                case 'r': val = 5f; break;
                case 'q': val = 9f; break;
                case 'k': val = 0f; break;
            }
            if (char.IsLower(ch)) val = -val; // black pieces lower-case in FEN

            int idx = sq % hiddenSize;
            vec[idx] += val / 10f;

            sq++;
        }

        // normalize
        float norm = 0f;
        foreach (var v in vec) norm += v * v;
        norm = (float)Math.Sqrt(norm) + 1e-8f;
        for (int i = 0; i < vec.Length; i++) vec[i] /= norm;

        return torch.tensor(vec).unsqueeze(0); // [1, hiddenSize]
    }

    // Plays a single self-play game between agent 0 (white) and agent 1 (black) inside MultiAgentFractalCore.
    // Returns list of chosen hidden states for each side and the result.
    private static (List<Tensor> whiteStates, List<bool> whiteCaptures, List<Tensor> blackStates, List<bool> blackCaptures, int result)
        PlayGame(MultiAgentFractalCore core, int hiddenSize, int maxMoves = 400)
    {
        var game = new ChessGame();

        var whiteStates = new List<Tensor>();
        var whiteCaptures = new List<bool>();
        var blackStates = new List<Tensor>();
        var blackCaptures = new List<bool>();

        int whiteCaptureCount = 0;
        int blackCaptureCount = 0;

        var hAgents = new Tensor[2];
        var cAgents = new Tensor[2];
        for (int i = 0; i < 2; i++)
        {
            hAgents[i] = torch.zeros(1, hiddenSize);
            cAgents[i] = torch.zeros(1, hiddenSize);
        }

        int moveCounter = 0;

        while (true)
        {
            var player = game.WhoseTurn; // ChessDotNet: WhoseTurn property

            var legal = game.GetValidMoves(player).ToArray();
            if (legal.Length == 0) break; // game over

            // Evaluate each legal move by simulating and encoding resulting board
            double bestScore = double.NegativeInfinity;
            Move bestMove = legal[0];

            int agentIndex = (player == Player.White) ? 0 : 1;

            foreach (var mv in legal)
            {
                var copy = new ChessGame(game.GetFen());
                copy.MakeMove(mv, true);
                var input = EncodeBoard(copy, hiddenSize);

                // evaluate with a fresh zero state to avoid mutating persistent state during search
                var h0 = torch.zeros(1, hiddenSize);
                var c0 = torch.zeros(1, hiddenSize);

                var (outTensor, hCand, cCand) = core.EvaluateAgent(agentIndex, input, h0, c0);

                var scoreTensor = outTensor.mean();
                double score = scoreTensor.ToSingle();
                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = mv;
                }

                // dispose temporaries
                outTensor.Dispose(); hCand.Dispose(); cCand.Dispose(); input.Dispose(); h0.Dispose(); c0.Dispose();
            }

            // detect capture on target square (if any)
            bool isCapture = false;
            try
            {
                // derive destination square from Move string (e.g. "G1-H3")
                var mvStr = bestMove.ToString();
                var matches = System.Text.RegularExpressions.Regex.Matches(mvStr, "[a-hA-H][1-8]");
                if (matches.Count > 0)
                {
                    var dest = matches[matches.Count - 1].Value.ToLowerInvariant();
                    try
                    {
                        var pos = new Position(dest);
                        var targetPiece = game.GetPieceAt(pos);
                        if (targetPiece != null) isCapture = true;
                    }
                    catch { /* ignore API differences */ }
                }
            }
            catch { isCapture = false; }

            // apply best move to real game
            game.MakeMove(bestMove, true);

            // print game state after move (best-effort)
            try
            {
                var fenAfter = game.GetFen();
                Console.WriteLine($"Move {moveCounter + 1}: {player} -> {bestMove}. FEN: {fenAfter}");
            }
            catch
            {
                Console.WriteLine($"Move {moveCounter + 1}: {player} -> {bestMove}.");
            }

            // compute new persistent state using the real board after move
            var curInput = EncodeBoard(game, hiddenSize);
            var (outReal, hOut, cOut) = core.EvaluateAgent(agentIndex, curInput, hAgents[agentIndex], cAgents[agentIndex]);

            // update persistent state and record
            if (agentIndex == 0)
            {
                hAgents[0].Dispose(); cAgents[0].Dispose();
                hAgents[0] = hOut; cAgents[0] = cOut;
                whiteStates.Add(hOut);
                whiteCaptures.Add(isCapture);
                if (isCapture) whiteCaptureCount++;
            }
            else
            {
                hAgents[1].Dispose(); cAgents[1].Dispose();
                hAgents[1] = hOut; cAgents[1] = cOut;
                blackStates.Add(hOut);
                blackCaptures.Add(isCapture);
                if (isCapture) blackCaptureCount++;
            }

            // cleanup
            outReal.Dispose(); curInput.Dispose();

            moveCounter++;

            // basic terminal detection: check for mate/stalemate via legal moves of opponent next
            var nextPlayer = game.WhoseTurn;
            var nextLegal = game.GetValidMoves(nextPlayer).ToArray();
            if (nextLegal.Length == 0) break;
            if (moveCounter >= maxMoves) break;
        }

        // determine result: 1 white win, -1 black win, 0 draw
        int result = 0;

        try
        {
            // Prefer using checkmate detection APIs if available
            bool whiteCheckmated = false;
            bool blackCheckmated = false;
            try { whiteCheckmated = game.IsCheckmated(Player.White); } catch { }
            try { blackCheckmated = game.IsCheckmated(Player.Black); } catch { }

            if (blackCheckmated) result = 1;
            else if (whiteCheckmated) result = -1;
            else
            {
                // if no checkmate, attempt material or draw heuristics: simple: compare capture counts
                if (whiteCaptureCount > blackCaptureCount) result = 1;
                else if (blackCaptureCount > whiteCaptureCount) result = -1;
                else result = 0;
            }
        }
        catch
        {
            result = 0;
        }

        return (whiteStates, whiteCaptures, blackStates, blackCaptures, result);
    }

    // Public entry point: run self-play training for a number of games.
    // Uses a shared MultiAgentFractalCore so its internal opponents learn from each other's moves.
    public static void SelfPlayTrain(int games = 10, int hiddenSize = 128, int maxDepth = 2, int maxMovesPerGame = 200)
    {
        Console.WriteLine($"Starting self-play training: games={games} hidden={hiddenSize} depth={maxDepth}");

        // create a shared core with two internal agents
        var core = new MultiAgentFractalCore(hiddenSize, maxDepth, 2);

        var parameters = core.parameters();
        var optimizer = torch.optim.Adam(parameters, lr: 1e-3);

        for (int g = 1; g <= games; g++)
        {
            Console.WriteLine($"=== Game {g} ===");
            var (whiteStates, whiteCaptures, blackStates, blackCaptures, result) = PlayGame(core, hiddenSize, maxMovesPerGame);

            // assemble loss
            var loss = torch.tensor(0f);

            float captureReward = 0.5f; // weight for immediate capture reward

            if (result == 1)
            {
                // white won: maximize whiteStates' means, minimize blackStates
                foreach (var h in whiteStates) { loss -= h.mean(); }
                foreach (var h in blackStates) { loss += h.mean(); }
                // encourage captures
                for (int i = 0; i < whiteCaptures.Count; i++) if (whiteCaptures[i]) loss -= captureReward * whiteStates[i].mean();
                for (int i = 0; i < blackCaptures.Count; i++) if (blackCaptures[i]) loss += captureReward * blackStates[i].mean();
            }
            else if (result == -1)
            {
                foreach (var h in blackStates) { loss -= h.mean(); }
                foreach (var h in whiteStates) { loss += h.mean(); }
                for (int i = 0; i < blackCaptures.Count; i++) if (blackCaptures[i]) loss -= captureReward * blackStates[i].mean();
                for (int i = 0; i < whiteCaptures.Count; i++) if (whiteCaptures[i]) loss += captureReward * whiteStates[i].mean();
            }
            else
            {
                // draw: small regularization to keep gradients mild, but still reward captures
                for (int i = 0; i < whiteStates.Count; i++)
                {
                    var h = whiteStates[i];
                    loss += 0.0f * h.mean();
                    if (whiteCaptures[i]) loss -= (captureReward * 0.01f) * h.mean();
                }
                for (int i = 0; i < blackStates.Count; i++)
                {
                    var h = blackStates[i];
                    loss += 0.0f * h.mean();
                    if (blackCaptures[i]) loss -= (captureReward * 0.01f) * h.mean();
                }
            }

            // perform optimization step if loss non-zero
            try
            {
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                Console.WriteLine($"Game {g} training step done. Loss={loss.ToSingle():F6} Result={result}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Training step failed: {ex.Message}");
            }

            // dispose stored tensors to avoid memory leaks
            foreach (var t in whiteStates) try { t.Dispose(); } catch { }
            foreach (var t in blackStates) try { t.Dispose(); } catch { }
        }

        Console.WriteLine("Self-play training finished.");
    }
}