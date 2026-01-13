/*using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// Sparse activation module: only apply linear transform to neurons with positive activation.
// Input shape: [1, hiddenSize]
public class SparseActivationModule : nn.Module
{
    private readonly Linear fc;
    private readonly int hiddenSize;

    public SparseActivationModule(int hiddenSize) : base("sparse_activation")
    {
        this.hiddenSize = hiddenSize;
        // Linear layer of shape (hiddenSize, hiddenSize)
        fc = nn.Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    // Forward takes x: [1, hiddenSize] and returns reconstructed tensor of same shape
    public Tensor forward(Tensor x)
    {
        // 1) Input: ensure we have the expected shape [1, H]
        // (no change, but keep for clarity)
        var input = x;

        // 2) Create boolean mask: mask = x > 0
        // mask has same shape [1, H] and marks active neurons
        var mask = input.gt(0);

        // 3) Extract active neuron values as a 1-D tensor
        // active will have shape [k] where k = number of true entries in mask
        var active = input.masked_select(mask);

        // 4) Prepare a tensor that contains only the active values at their original positions
        //    and zeros elsewhere. This lets us call the full Linear(hiddenSize, hiddenSize)
        //    but with zeros for inactive neurons so they produce no contribution.
        var inputForLinear = torch.zeros_like(input);
        // Fill the positions where mask==true with the active values
        inputForLinear = inputForLinear.masked_scatter(mask, active);

        // 5) Apply the linear layer to the prepared tensor
        //    This computes a transformed [1, H] tensor. Because inactive positions were zero,
        //    the transformation's output depends only on the active inputs.
        var transformed = fc.forward(inputForLinear);

        // 6) Scatter the transformed values back into the original positions.
        //    We want the output to retain original (unchanged) values for inactive positions
        //    and use the transformed values for active positions.
        var output = input.clone(); // start from original
        // Extract transformed entries at active positions and copy them back into output
        var transformedActive = transformed.masked_select(mask);
        output = output.masked_scatter(mask, transformedActive);

        // 7) Return reconstructed tensor of shape [1, H]
        return output;
    }
}
*/