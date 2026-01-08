import torch
import matplotlib.pyplot as plt
from src.data import get_loaders
from src.models import GrokkingMLP, GrokkingSNN
from src.train import train_model
from src.visualize_hidden import plot_hidden_comparison, plot_grokking_comparison
import os

def run_experiment(p=113, train_frac=0.3, num_epochs=15000, spike_lambda=0.01, device="cpu"):
    """
    Run grokking experiment comparing MLP and Sparse SNN.
    
    Args:
        p: Modulus for modular addition task
        train_frac: Fraction of data for training
        num_epochs: Number of training epochs
        spike_lambda: Spike count regularization weight (IEEE paper approach)
                     Higher values = fewer spikes = sparser SNN
        device: 'cpu' or 'cuda'
    """
    print(f"Starting Grokking Experiment with p={p}, train_frac={train_frac}")
    print(f"SNN spike regularization lambda: {spike_lambda}")
    
    train_loader, test_loader = get_loaders(p=p, train_frac=train_frac)
    
    # Define snapshot epochs for hidden layer visualization
    # Capture before, during, and after expected grokking
    snapshot_epochs = [0, 500, 1000, 2000, 3000, 5000, 7000]
    
    # 1. Train MLP
    print("\nTraining MLP...")
    mlp = GrokkingMLP(p=p, hidden_dim=128)
    mlp_history = train_model(
        mlp, train_loader, test_loader, 
        num_epochs=num_epochs, weight_decay=1.0, 
        snapshot_epochs=snapshot_epochs, device=device
    )
    
    # Save MLP model
    os.makedirs("models", exist_ok=True)
    torch.save(mlp.state_dict(), "models/mlp_final.pth")
    print("Saved MLP model to models/mlp_final.pth")
    
    # 2. Train SNN with spike sparsity regularization (IEEE paper approach)
    print("\nTraining Sparse SNN...")
    snn_model = GrokkingSNN(p=p, hidden_dim=128, num_steps=15)
    snn_history = train_model(
        snn_model, train_loader, test_loader, 
        num_epochs=num_epochs, weight_decay=1.0,
        spike_lambda=spike_lambda, snapshot_epochs=snapshot_epochs,
        device=device
    )
    
    # Save SNN model
    torch.save(snn_model.state_dict(), "models/snn_final.pth")
    print("Saved SNN model to models/snn_final.pth")

    # 3. Generate visualizations
    os.makedirs("plots", exist_ok=True)
    
    # Basic training curves
    plot_results(mlp_history, snn_history)
    
    # Comprehensive grokking comparison (includes spike rate analysis)
    plot_grokking_comparison(mlp_history, snn_history, save_dir="plots")
    
    # Hidden layer pattern comparison across epochs
    plot_hidden_comparison(mlp_history, snn_history, save_dir="plots")
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print("Generated visualizations in plots/:")
    print("  - comparison.png (training curves)")
    print("  - grokking_comparison.png (comprehensive analysis)")
    print("  - activation_dist_epoch_*.png (activation distributions)")
    print("  - hidden_pca_epoch_*.png (PCA of hidden states)")
    print("  - neuron_heatmap_epoch_*.png (neuron activity patterns)")
    if snn_history.get("spike_rate"):
        print("  - spike_rate_evolution.png (SNN sparsity over training)")
    print("="*50)

def plot_results(mlp_history, snn_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Plot
    ax1.plot(mlp_history["train_acc"], label="MLP Train", linestyle="--", color="blue", alpha=0.5)
    ax1.plot(mlp_history["test_acc"], label="MLP Test", color="blue")
    ax1.plot(snn_history["train_acc"], label="SNN Train", linestyle="--", color="red", alpha=0.5)
    ax1.plot(snn_history["test_acc"], label="SNN Test", color="red")
    ax1.set_title("Accuracy vs Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.set_ylim(0, 105)
    
    # Loss Plot (Log Scale)
    ax2.semilogy(mlp_history["train_loss"], label="MLP Train", linestyle="--", color="blue", alpha=0.5)
    ax2.semilogy(mlp_history["test_loss"], label="MLP Test", color="blue")
    ax2.semilogy(snn_history["train_loss"], label="SNN Train", linestyle="--", color="red", alpha=0.5)
    ax2.semilogy(snn_history["test_loss"], label="SNN Test", color="red")
    ax2.set_title("Loss vs Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("plots/comparison.png")
    print("\nResults saved to plots/comparison.png")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Full experiment with IEEE paper's spike sparsity regularization
    # spike_lambda controls the sparsity (higher = fewer spikes)
    # The paper reports ~70% spike reduction at iso-accuracy
    run_experiment(
        p=113, 
        train_frac=0.4, 
        num_epochs=10000, 
        spike_lambda=0.01,  # IEEE paper approach
        device=device
    )

