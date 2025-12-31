import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, test_loader, num_epochs=2000, lr=1e-3, weight_decay=1.0, device="cpu"):
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
    
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
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
        
        pbar.set_description(f"Train Acc: {history['train_acc'][-1]:.1f}% | Test Acc: {history['test_acc'][-1]:.1f}%")
        
        # Early exit if we reached 100% test accuracy (optional)
        # if history["test_acc"][-1] > 99.9:
        #     break
            
    return history

