"""
Hidden Layer Visualization for MLP vs SNN Grokking Comparison.

Compares hidden layer activation patterns between MLP (ReLU activations) 
and SNN (spike rates) to understand if similar structures emerge during grokking.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def extract_all_hidden_activations(model, data_loader, device="cpu"):
    """
    Extract hidden layer activations for all samples in the data loader.
    
    Returns:
        activations: [N, hidden_dim] tensor of hidden layer states
        labels: [N] tensor of corresponding labels
    """
    model.eval()
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            act = model.get_hidden_activations(x)
            all_activations.append(act.cpu())
            all_labels.append(y)
    
    return torch.cat(all_activations), torch.cat(all_labels)


def plot_activation_distribution(mlp_act, snn_act, epoch, save_path=None):
    """
    Plot histogram of activation values for MLP vs SNN.
    
    Args:
        mlp_act: MLP hidden activations [N, hidden_dim]
        snn_act: SNN spike counts [N, hidden_dim]
        epoch: Training epoch for title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MLP ReLU activations
    mlp_flat = mlp_act.flatten().numpy()
    axes[0].hist(mlp_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title(f"MLP Hidden Activations (Epoch {epoch})")
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Calculate sparsity (% of zeros)
    mlp_sparsity = (mlp_flat == 0).sum() / len(mlp_flat) * 100
    axes[0].text(0.95, 0.95, f"Sparsity: {mlp_sparsity:.1f}%", 
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # SNN spike rates
    snn_flat = snn_act.flatten().numpy()
    axes[1].hist(snn_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"SNN Spike Counts (Epoch {epoch})")
    axes[1].set_xlabel("Spike Count (over time steps)")
    axes[1].set_ylabel("Count")
    
    # Calculate sparsity (% of no spikes)
    snn_sparsity = (snn_flat == 0).sum() / len(snn_flat) * 100
    axes[1].text(0.95, 0.95, f"Sparsity: {snn_sparsity:.1f}%", 
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_hidden_pca(mlp_act, snn_act, mlp_labels, snn_labels, epoch, p=113, save_path=None):
    """
    PCA visualization of hidden representations colored by output class.
    Shows if similar clustering emerges in MLP vs SNN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use a subset of classes for clear visualization (first 10 classes)
    num_classes_to_show = min(10, p)
    
    # MLP PCA
    pca = PCA(n_components=2)
    mlp_pca = pca.fit_transform(mlp_act.numpy())
    
    for c in range(num_classes_to_show):
        mask = mlp_labels.numpy() == c
        if mask.sum() > 0:
            axes[0].scatter(mlp_pca[mask, 0], mlp_pca[mask, 1], 
                          label=f'{c}', alpha=0.6, s=10)
    axes[0].set_title(f"MLP Hidden Layer PCA (Epoch {epoch})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    # SNN PCA
    snn_pca = pca.fit_transform(snn_act.numpy())
    
    for c in range(num_classes_to_show):
        mask = snn_labels.numpy() == c
        if mask.sum() > 0:
            axes[1].scatter(snn_pca[mask, 0], snn_pca[mask, 1], 
                          label=f'{c}', alpha=0.6, s=10)
    axes[1].set_title(f"SNN Hidden Layer PCA (Epoch {epoch})")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_neuron_activity_heatmap(mlp_act, snn_act, epoch, save_path=None):
    """
    Heatmap showing which neurons are active for different inputs.
    Rows = samples, Columns = neurons.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Take first 50 samples for visualization
    n_samples = min(50, mlp_act.shape[0])
    
    # MLP heatmap
    im1 = axes[0].imshow(mlp_act[:n_samples].numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title(f"MLP Neuron Activity (Epoch {epoch})")
    axes[0].set_xlabel("Neuron Index")
    axes[0].set_ylabel("Sample Index")
    plt.colorbar(im1, ax=axes[0], label="Activation")
    
    # SNN heatmap
    im2 = axes[1].imshow(snn_act[:n_samples].numpy(), aspect='auto', cmap='hot')
    axes[1].set_title(f"SNN Spike Activity (Epoch {epoch})")
    axes[1].set_xlabel("Neuron Index")
    axes[1].set_ylabel("Sample Index")
    plt.colorbar(im2, ax=axes[1], label="Spike Count")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_hidden_comparison(mlp_history, snn_history, save_dir="plots"):
    """
    Generate comprehensive hidden layer comparison plots from training histories.
    
    Args:
        mlp_history: Training history from MLP containing hidden_snapshots
        snn_history: Training history from SNN containing hidden_snapshots
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Find common epochs
    mlp_epochs = set(mlp_history.get("hidden_snapshots", {}).keys())
    snn_epochs = set(snn_history.get("hidden_snapshots", {}).keys())
    common_epochs = sorted(mlp_epochs & snn_epochs)
    
    if not common_epochs:
        print("No common snapshot epochs found. Skipping hidden layer visualization.")
        return
    
    print(f"\nGenerating hidden layer visualizations for epochs: {common_epochs}")
    
    for epoch in common_epochs:
        mlp_snap = mlp_history["hidden_snapshots"][epoch]
        snn_snap = snn_history["hidden_snapshots"][epoch]
        
        mlp_act = mlp_snap["activations"]
        snn_act = snn_snap["activations"]
        mlp_labels = mlp_snap["labels"]
        snn_labels = snn_snap["labels"]
        
        # 1. Activation distribution
        plot_activation_distribution(
            mlp_act, snn_act, epoch,
            save_path=f"{save_dir}/activation_dist_epoch_{epoch}.png"
        )
        
        # 2. PCA visualization
        plot_hidden_pca(
            mlp_act, snn_act, mlp_labels, snn_labels, epoch,
            save_path=f"{save_dir}/hidden_pca_epoch_{epoch}.png"
        )
        
        # 3. Neuron activity heatmap
        plot_neuron_activity_heatmap(
            mlp_act, snn_act, epoch,
            save_path=f"{save_dir}/neuron_heatmap_epoch_{epoch}.png"
        )
    
    # 4. Plot spike rate over training (SNN only)
    if snn_history.get("spike_rate"):
        plot_spike_rate_evolution(snn_history["spike_rate"], save_dir)
    
    print(f"Hidden layer visualizations saved to {save_dir}/")


def plot_spike_rate_evolution(spike_rates, save_dir="plots"):
    """Plot how SNN spike rate changes during training."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(spike_rates, color='red', alpha=0.8)
    ax.set_title("SNN Spike Rate During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Spike Rate (normalized)")
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(spike_rates) > 10:
        z = np.polyfit(range(len(spike_rates)), spike_rates, 1)
        p = np.poly1d(z)
        ax.plot(p(range(len(spike_rates))), '--', color='darkred', 
                label=f'Trend (slope: {z[0]:.2e})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/spike_rate_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spike rate evolution plot saved.")


def plot_grokking_comparison(mlp_history, snn_history, save_dir="plots"):
    """
    Compare grokking dynamics between MLP and SNN.
    Highlights the epoch at which each model achieves 95%+ test accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Find grokking points (95% test accuracy)
    mlp_grok_epoch = None
    snn_grok_epoch = None
    
    for i, acc in enumerate(mlp_history["test_acc"]):
        if acc >= 95 and mlp_grok_epoch is None:
            mlp_grok_epoch = i
            break
    
    for i, acc in enumerate(snn_history["test_acc"]):
        if acc >= 95 and snn_grok_epoch is None:
            snn_grok_epoch = i
            break
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    ax.plot(mlp_history["train_acc"], label="MLP Train", linestyle="--", color="blue", alpha=0.5)
    ax.plot(mlp_history["test_acc"], label="MLP Test", color="blue")
    ax.plot(snn_history["train_acc"], label="SNN Train", linestyle="--", color="red", alpha=0.5)
    ax.plot(snn_history["test_acc"], label="SNN Test", color="red")
    
    if mlp_grok_epoch:
        ax.axvline(x=mlp_grok_epoch, color='blue', linestyle=':', alpha=0.5)
        ax.text(mlp_grok_epoch, 50, f'MLP: {mlp_grok_epoch}', color='blue', fontsize=9)
    if snn_grok_epoch:
        ax.axvline(x=snn_grok_epoch, color='red', linestyle=':', alpha=0.5)
        ax.text(snn_grok_epoch, 40, f'SNN: {snn_grok_epoch}', color='red', fontsize=9)
    
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.3, label='95% threshold')
    ax.set_title("Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.set_ylim(0, 105)
    
    # 2. Loss comparison
    ax = axes[0, 1]
    ax.semilogy(mlp_history["train_loss"], label="MLP Train", linestyle="--", color="blue", alpha=0.5)
    ax.semilogy(mlp_history["test_loss"], label="MLP Test", color="blue")
    ax.semilogy(snn_history["train_loss"], label="SNN Train", linestyle="--", color="red", alpha=0.5)
    ax.semilogy(snn_history["test_loss"], label="SNN Test", color="red")
    ax.set_title("Loss Comparison (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    
    # 3. Spike rate evolution (SNN)
    ax = axes[1, 0]
    if snn_history.get("spike_rate"):
        ax.plot(snn_history["spike_rate"], color='red', alpha=0.8)
        ax.set_title("SNN Spike Rate During Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Spike Rate")
        ax.grid(True, alpha=0.3)
        
        if snn_grok_epoch:
            ax.axvline(x=snn_grok_epoch, color='green', linestyle=':', label='Grokking point')
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No spike rate data\n(spike_lambda=0)", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title("SNN Spike Rate")
    
    # 4. Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = [
        "GROKKING COMPARISON SUMMARY",
        "=" * 40,
        f"MLP Grokking Epoch: {mlp_grok_epoch if mlp_grok_epoch else 'Not reached'}",
        f"SNN Grokking Epoch: {snn_grok_epoch if snn_grok_epoch else 'Not reached'}",
        "",
    ]
    
    if mlp_grok_epoch and snn_grok_epoch:
        speedup = (mlp_grok_epoch - snn_grok_epoch) / mlp_grok_epoch * 100
        if speedup > 0:
            summary.append(f"SNN is {speedup:.1f}% FASTER than MLP")
        else:
            summary.append(f"MLP is {-speedup:.1f}% FASTER than SNN")
    
    summary.extend([
        "",
        f"Final MLP Test Acc: {mlp_history['test_acc'][-1]:.1f}%",
        f"Final SNN Test Acc: {snn_history['test_acc'][-1]:.1f}%",
    ])
    
    if snn_history.get("spike_rate"):
        initial_spk = snn_history["spike_rate"][0]
        final_spk = snn_history["spike_rate"][-1]
        reduction = (initial_spk - final_spk) / initial_spk * 100
        summary.extend([
            "",
            f"Initial SNN Spike Rate: {initial_spk:.4f}",
            f"Final SNN Spike Rate: {final_spk:.4f}",
            f"Spike Reduction: {reduction:.1f}%"
        ])
    
    ax.text(0.1, 0.9, '\n'.join(summary), transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/grokking_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grokking comparison saved to {save_dir}/grokking_comparison.png")


if __name__ == "__main__":
    # Test module with dummy data
    print("visualize_hidden.py module loaded successfully.")
