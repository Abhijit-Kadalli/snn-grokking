import torch
import matplotlib.pyplot as plt
from src.data import get_loaders
from src.models import GrokkingMLP, GrokkingSNN
from src.train import train_model
import os

def run_experiment(p=113, train_frac=0.3, num_epochs=15000, device="cpu"):
    print(f"Starting Grokking Experiment with p={p}, train_frac={train_frac}")
    
    train_loader, test_loader = get_loaders(p=p, train_frac=train_frac)
    
    # 1. Train MLP
    print("\nTraining MLP...")
    mlp = GrokkingMLP(p=p, hidden_dim=128)
    mlp_history = train_model(mlp, train_loader, test_loader, num_epochs=num_epochs, weight_decay=1.0, device=device)
    
    # 2. Train SNN
    print("\nTraining SNN...")
    snn_model = GrokkingSNN(p=p, hidden_dim=128, num_steps=15)
    snn_history = train_model(snn_model, train_loader, test_loader, num_epochs=num_epochs, weight_decay=1.0, device=device)
    
    # Save the trained SNN model
    torch.save(snn_model.state_dict(), "models/snn_final.pth")
    print("\nSaved SNN model to models/snn_final.pth")

    # 3. Plot Results
    plot_results(mlp_history, snn_history)

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
    # Reducing epochs for a quick verification if needed, 
    # but the proposal asks for the full experiment.
    # We will use 10k epochs as a baseline.
    run_experiment(p=113, train_frac=0.4, num_epochs=10000, device=device)
