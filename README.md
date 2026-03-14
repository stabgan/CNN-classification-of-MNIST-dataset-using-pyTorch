# CNN Classification of MNIST using PyTorch

A minimal two-layer convolutional neural network that classifies handwritten digits (0–9) from the MNIST dataset using PyTorch.

## What It Does

Trains a CNN on the 60,000-image MNIST training set and evaluates accuracy on the 10,000-image test set every 500 iterations. Automatically uses CUDA when available.

## Architecture

```
Input (1×28×28)
  → Conv2d(1→16, 5×5) → ReLU → MaxPool2d(2×2)   # 16×12×12
  → Conv2d(16→32, 5×5) → ReLU → MaxPool2d(2×2)   # 32×4×4
  → Flatten (512)
  → Linear(512→10)
  → Output (10 classes)
```

| Hyperparameter | Value |
|---|---|
| Batch size | 100 |
| Iterations | 3,000 |
| Epochs | 5 |
| Learning rate | 0.01 |
| Optimizer | SGD |
| Loss | CrossEntropyLoss |

## 🛠 Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🐍 | Python 3.x | Language |
| 🔥 | PyTorch | Deep learning framework |
| 🖼️ | torchvision | MNIST dataset & transforms |

## Installation

```bash
pip install torch torchvision
```

## Usage

```bash
python cnn.py
```

MNIST data is downloaded automatically to `./data/` on first run.

## License

MIT
