import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def make_data(p=113, train_frac=0.3, seed=42):
    """
    Generates modular addition data for a + b % p.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate all pairs (a, b)
    all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = (all_pairs[:, 0] + all_pairs[:, 1]) % p
    
    # Shuffle
    indices = torch.randperm(len(all_pairs))
    all_pairs = all_pairs[indices]
    labels = labels[indices]
    
    # Split
    split_idx = int(len(all_pairs) * train_frac)
    
    train_data = all_pairs[:split_idx]
    train_labels = labels[:split_idx]
    
    test_data = all_pairs[split_idx:]
    test_labels = labels[split_idx:]
    
    return train_data, train_labels, test_data, test_labels

class ModularDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_loaders(p=113, train_frac=0.3, batch_size=512):
    train_data, train_labels, test_data, test_labels = make_data(p, train_frac)
    
    train_ds = ModularDataset(train_data, train_labels)
    test_ds = ModularDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    
    return train_loader, test_loader

