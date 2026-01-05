import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.models import GrokkingSNN
from src.data import make_data

def visualize_snn_dynamics(model_path="models/snn_final.pth", p=113, device="cpu"):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please run main.py first to train the model.")
        return

    # Load Model
    model = GrokkingSNN(p=p, hidden_dim=128, num_steps=15)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get a sample input
    # Let's find a simple case, e.g., 50 + 63 = 0 (mod 113) or just random
    # We'll just pick a random one
    a, b = 10, 20
    target = (a + b) % p
    
    x = torch.tensor([[a, b]], device=device)
    
    print(f"Visualizing for input: {a} + {b} = {target} (mod {p})")
    
    with torch.no_grad():
        spk1, mem1, spk2, mem2 = model(x, return_dynamics=True)
        
    # Tensors are [num_steps, batch, features] -> [num_steps, features]
    spk1 = spk1.squeeze(1).cpu().numpy()
    mem1 = mem1.squeeze(1).cpu().numpy()
    spk2 = spk2.squeeze(1).cpu().numpy()
    mem2 = mem2.squeeze(1).cpu().numpy()
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Raster Plot of Hidden Layer Spikes
    # spk1 is [steps, hidden_dim]
    # We want indices where spike is 1
    spike_times, neuron_indices = np.where(spk1 > 0)
    axes[0].scatter(spike_times, neuron_indices, s=2, c='black')
    axes[0].set_title(f"Hidden Layer Spikes (Input: {a}+{b})")
    axes[0].set_ylabel("Neuron Index")
    axes[0].set_xlim(-0.5, model.num_steps - 0.5)
    
    # 2. Output Layer Membrane Potentials (Target vs Others)
    # mem2 is [steps, p]
    steps = np.arange(model.num_steps)
    
    # Plot target class
    axes[1].plot(steps, mem2[:, target], label=f"Target Class ({target})", color="green", linewidth=2)
    
    # Plot a few random other classes
    other_indices = np.random.choice([i for i in range(p) if i != target], 5, replace=False)
    for idx in other_indices:
        axes[1].plot(steps, mem2[:, idx], label=f"Class {idx}", alpha=0.3)
        
    axes[1].set_title("Output Layer Membrane Potentials")
    axes[1].set_ylabel("Membrane Potential")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Hidden Layer Membrane Potentials Heatmap
    im = axes[2].imshow(mem1.T, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
    axes[2].set_title("Hidden Layer Membrane Potentials")
    axes[2].set_ylabel("Neuron Index")
    axes[2].set_xlabel("Time Step")
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("plots/snn_visualization.png")
    print("Visualization saved to plots/snn_visualization.png")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visualize_snn_dynamics(device=device)

