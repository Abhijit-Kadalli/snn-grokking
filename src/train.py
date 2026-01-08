import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, test_loader, num_epochs=2000, lr=1e-3, 
                weight_decay=1.0, spike_lambda=0.0, snapshot_epochs=None, device="cpu"):
    """
    Train a model with optional spike sparsity regularization.
    
    Args:
        model: GrokkingMLP or GrokkingSNN instance
        spike_lambda: Weight for spike count regularization (IEEE paper approach)
                     Set > 0 for SNNs to encourage sparse spiking
        snapshot_epochs: List of epochs at which to record hidden activations
    """
    model.to(device)
    
    # Separate parameters for weight decay
    # We do NOT want to decay SNN dynamics parameters (beta, threshold)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if "beta" in name or "threshold" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # Check if model supports spike count tracking (SNN)
    is_snn = hasattr(model, 'forward') and spike_lambda > 0
    
    # Default snapshot epochs if not provided
    if snapshot_epochs is None:
        snapshot_epochs = [0, 1000, 3000, 5000, 7000]
    
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "spike_rate": [],  # Average spike rate per epoch (SNN only)
        "hidden_snapshots": {}  # Activation snapshots at key epochs
    }
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        epoch_spike_rate = 0.0
        batch_count = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Apply spike count regularization for SNNs (IEEE paper approach)
            if is_snn:
                outputs, spike_rate = model(x, return_spike_count=True)
                sparsity_loss = spike_lambda * spike_rate
                loss = criterion(outputs, y) + sparsity_loss
                epoch_spike_rate += spike_rate.item()
                batch_count += 1
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(100. * correct / total)
        
        # Track spike rate for SNNs
        if is_snn and batch_count > 0:
            history["spike_rate"].append(epoch_spike_rate / batch_count)
        
        # Capture hidden activation snapshots at key epochs
        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                # Get a fixed sample from test loader for consistent comparison
                sample_x, sample_y = next(iter(test_loader))
                sample_x = sample_x[:100].to(device)  # Use first 100 samples
                sample_y = sample_y[:100].to(device)
                activations = model.get_hidden_activations(sample_x)
                history["hidden_snapshots"][epoch] = {
                    "activations": activations.cpu(),
                    "labels": sample_y.cpu()
                }
            model.train()
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                
        history["test_loss"].append(test_loss / len(test_loader))
        history["test_acc"].append(100. * correct / total)
        
        # Update progress bar with spike rate info for SNNs
        if is_snn and history["spike_rate"]:
            pbar.set_description(f"Train: {history['train_acc'][-1]:.1f}% | Test: {history['test_acc'][-1]:.1f}% | Spk: {history['spike_rate'][-1]:.3f}")
        else:
            pbar.set_description(f"Train Acc: {history['train_acc'][-1]:.1f}% | Test Acc: {history['test_acc'][-1]:.1f}%")
        
        # Early exit if we reached 100% test accuracy (optional)
        # if history["test_acc"][-1] > 99.9:
        #     break
            
    return history

