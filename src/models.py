import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

class GrokkingMLP(nn.Module):
    def __init__(self, p, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(p, hidden_dim)
        # We concatenate the embeddings of a and b
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, p)
        
    def forward(self, x):
        # x is [batch, 2]
        embedded = self.embed(x) # [batch, 2, hidden_dim]
        embedded = embedded.view(x.size(0), -1) # [batch, 2 * hidden_dim]
        h = self.relu(self.linear1(embedded))
        out = self.linear2(h)
        return out

class GrokkingSNN(nn.Module):
    def __init__(self, p, hidden_dim=128, num_steps=15, beta=0.9, threshold=1.0):
        super().__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        self.embed = nn.Embedding(p, hidden_dim)
        
        # SNN Layers with learnable parameters
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, 
                              learn_beta=True, learn_threshold=True, 
                              spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(hidden_dim, p)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, 
                              learn_beta=True, learn_threshold=True, 
                              spike_grad=spike_grad, reset_mechanism="none") # None for output layer potential

    def forward(self, x, return_dynamics=False):
        # Clamp parameters to valid ranges to prevent instability
        with torch.no_grad():
            self.lif1.beta.clamp_(0.1, 0.9)
            self.lif2.beta.clamp_(0.1, 0.9)
            self.lif1.threshold.clamp_(min=0.1)
            self.lif2.threshold.clamp_(min=0.1)

        # x is [batch, 2]
        # Input Coding: Constant Current Injection
        # The embedding is repeated across all time steps
        embedded = self.embed(x) # [batch, 2, hidden_dim]
        embedded = embedded.view(x.size(0), -1) # [batch, 2 * hidden_dim]
        
        # Initialize membrane potential
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record dynamics
        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(embedded)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            if return_dynamics:
                spk1_rec.append(spk1)
                mem1_rec.append(mem1)
                spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        if return_dynamics:
            return torch.stack(spk1_rec), torch.stack(mem1_rec), torch.stack(spk2_rec), torch.stack(mem2_rec)

        # Output Decoding: Last Membrane Potential (Temporal/Potential coding)
        # This provides a smoother gradient than rate-based spike counting
        return mem2_rec[-1]

