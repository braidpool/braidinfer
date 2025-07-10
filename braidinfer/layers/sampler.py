import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, 
                top_k: torch.Tensor = None, top_p: torch.Tensor = None, 
                min_p: torch.Tensor = None):
        """
        Sample from logits with temperature, top_k, top_p, and min_p filtering.
        
        Args:
            logits: [batch_size, vocab_size] tensor of logits
            temperatures: [batch_size] tensor of temperatures
            top_k: [batch_size] tensor of top_k values (0 means no limit)
            top_p: [batch_size] tensor of top_p values (1.0 means no nucleus sampling)
            min_p: [batch_size] tensor of min_p values (0.0 means no min-p filtering)
            
        Returns:
            Sampled token indices [batch_size]
        """
        batch_size, vocab_size = logits.shape
        logits = logits.to(torch.float)
        
        # Apply temperature
        logits_scaled = logits / temperatures.unsqueeze(dim=1).clamp(min=1e-10)
        
        # For greedy decoding (temperature == 0)
        greedy_tokens = logits.argmax(dim=-1)
        
        # Convert to probabilities
        probs = torch.softmax(logits_scaled, dim=-1, dtype=torch.float)
        
        # Apply min_p filtering first (before top_k/top_p)
        if min_p is not None:
            # Calculate the threshold for each sequence
            max_probs = probs.max(dim=-1, keepdim=True).values
            min_p_thresholds = max_probs * min_p.unsqueeze(1)
            
            # Set probabilities below threshold to 0
            probs = torch.where(probs >= min_p_thresholds, probs, torch.zeros_like(probs))
        
        # Apply top_k filtering
        if top_k is not None:
            # For each sequence, keep only top_k tokens
            for i in range(batch_size):
                k = int(top_k[i].item())
                if k > 0 and k < vocab_size:
                    # Get top_k indices
                    topk_probs, topk_indices = probs[i].topk(k)
                    # Create mask for top_k tokens
                    mask = torch.zeros_like(probs[i], dtype=torch.bool)
                    mask[topk_indices] = True
                    # Zero out non-top_k probabilities
                    probs[i] = torch.where(mask, probs[i], torch.zeros_like(probs[i]))
        
        # Apply top_p (nucleus) filtering
        if top_p is not None:
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=-1)
            
            # Find where cumsum exceeds top_p
            for i in range(batch_size):
                p = top_p[i].item()
                if p < 1.0:
                    # Find cutoff index
                    cutoff_idx = (cumsum_probs[i] > p).nonzero(as_tuple=True)[0]
                    if len(cutoff_idx) > 0:
                        cutoff_idx = cutoff_idx[0].item()
                        # Keep at least one token
                        cutoff_idx = max(1, cutoff_idx)
                        # Create mask for nucleus tokens
                        nucleus_mask = torch.zeros_like(probs[i], dtype=torch.bool)
                        nucleus_mask[sorted_indices[i, :cutoff_idx]] = True
                        # Zero out non-nucleus probabilities
                        probs[i] = torch.where(nucleus_mask, probs[i], torch.zeros_like(probs[i]))
        
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        
        # Sample from the filtered distribution
        epsilon = 1e-10
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        
        # Use greedy tokens where temperature is 0
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)