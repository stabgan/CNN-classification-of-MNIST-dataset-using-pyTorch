# CNN Classification of MNIST using PyTorch

A minimal convolutional neural network that classifies handwritten digits (0–9) from the MNIST dataset using PyTorch.

## What It Does

Trains a two-layer CNN on the 60 000-image MNIST training set and evaluates accuracy on the 10 000-image test set. Logs loss and accuracy every 500 iterations. Supports CUDA if available.

## Architecture

```
Input (1×28×28)
  → Conv2d(1→16, 5×5) → ReLU → MaxPool2d(2×2)    # output: 16×12×12
  → Conv2d(16→32, 5×5) → ReLU → MaxPool2d(2×2)    # output: 32×4×4
  → Flatten (512)
  → Linear(512→10)
  → Output (10 classes)
```

| Hyperparameter | Value |
|---|---|
| Batch size | 100 |
| Iterations | 3 000 |
| Epochs | 5 |
| Learning rate | 0.01 |
| Optimizer | SGD |
| Loss | CrossEntropyLoss |

## Dependencies

- Python 3.x
- PyTorch
- torchvision

```bash
pip install torch torchvision
```

## Usage

```bash
python cnn.py
```

MNIST data is downloaded automatically to `./data/` on first run.

## Known Bugs & Deprecations

1. **`torch.autograd.Variable` is deprecated.** Since PyTorch 0.4+, tensors track gradients natively. All `Variable(...)` wrapping is unnecessary and should be removed.

2. **`loss.data[0]` crashes on modern PyTorch.** Zero-dimensional tensors no longer support indexing. Replace with `loss.item()`.

3. **Comment says reshape size is `(100, 32, 7, 7)` — it's actually `(batch, 32, 4, 4)`.** After two 5×5 convolutions (no padding) and two 2×2 max-pools on a 28×28 input, the spatial dimensions are 4×4, not 7×7. The `view` call and the `Linear` layer use the correct value (512); only the comment is wrong.

## License

MIT
