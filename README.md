# 🧠 CNN Classification of MNIST using PyTorch

A minimal convolutional neural network that classifies handwritten digits (0–9) from the MNIST dataset, built with PyTorch.

Trains on the 60,000-image MNIST training set and evaluates accuracy on the 10,000-image test set. Logs loss and accuracy every 500 iterations. Automatically uses CUDA when available.

---

## 🏗️ Architecture

```
Input (1x28x28)
  -> Conv2d(1->16, 5x5) -> ReLU -> MaxPool2d(2x2)    # -> 16x12x12
  -> Conv2d(16->32, 5x5) -> ReLU -> MaxPool2d(2x2)    # -> 32x4x4
  -> Flatten (512)
  -> Linear(512->10)
  -> Output (10 classes)
```

| Hyperparameter | Value |
|---|---|
| Batch size | 100 |
| Iterations | 3,000 |
| Epochs | 5 |
| Learning rate | 0.01 |
| Optimizer | SGD |
| Loss function | CrossEntropyLoss |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🐍 Python 3.x | Language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ torchvision | MNIST dataset and transforms |
| 🚀 CUDA | GPU acceleration (optional) |

---

## 📦 Dependencies

- Python >= 3.8
- PyTorch >= 1.0
- torchvision

```bash
pip install torch torchvision
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/stabgan/CNN-classification-of-MNIST-dataset-using-pyTorch.git
cd CNN-classification-of-MNIST-dataset-using-pyTorch

# Install dependencies
pip install torch torchvision

# Train and evaluate
python cnn.py
```

MNIST data is downloaded automatically to `./data/` on first run.

---

## ⚠️ Known Issues

- The model is intentionally simple (two conv layers + one FC layer) and is meant as a learning example, not a production classifier.
- No learning rate scheduler or data augmentation is used.
- Training hyperparameters are hardcoded; consider adding `argparse` for configurability.

---

## 📄 License

MIT
