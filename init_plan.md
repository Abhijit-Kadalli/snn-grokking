# Spike-Grok Experiment Plan

This plan implements the "Do Spiking Networks Speedrun Grokking?" research proposal. We will compare a standard MLP against an SNN on the modular addition task $a + b \pmod p$.

## 1. Environment & Dependencies

- Update `pyproject.toml` with required libraries:
- `torch`, `snntorch`: For modeling.
- `numpy`, `matplotlib`, `tqdm`: For data and visualization.
- `einops`: For tensor manipulation.

## 2. Data Generation (`src/data.py`)

- Implement `make_data(p, train_frac)`:
- Generate all pairs $(a, b)$ for modulo $p$ (e.g., $p=113$).
- Split into Train (e.g., 30-50%) and Test sets.
- Returns PyTorch tensors/DataLoaders.

## 3. Models (`src/models.py`)

- **Baseline MLP (`GrokkingMLP`)**:
- Architecture: Embedding $\to$ Hidden Layer (ReLU) $\to$ Output.
- Standard architecture used in grokking papers.
- **Spiking Network (`GrokkingSNN`)**:
- Architecture: Embedding $\to$ Spiking Hidden Layer (LIF) $\to$ Output (Integrate).
- Uses `snntorch` for Leaky Integrate-and-Fire neurons and surrogate gradients.
- Time-stepped execution to allow for temporal coding.

## 4. Training Engine (`src/train.py`)

- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: `AdamW` with significant weight decay (key driver for grokking).
- **Loop**:
- Train for large number of epochs (e.g., 10k-20k).
- Track: Train Accuracy, Test Accuracy, Train Loss, Test Loss.
- Store results for plotting.

## 5. Execution & Analysis (`main.py`)

- **Experiment 1: The Speed Test**
- Train MLP and SNN on the exact same split.
- Compare the "epochs to generalization" (when test accuracy spikes).
- **Visualization**:
- Plot Accuracy vs Epochs for both models.
- Save plots to `plots/`.

## 6. (Future) Mechanism Analysis

- Once the speed test is working, we will implement Fourier analysis to check for rate vs. phase coding.