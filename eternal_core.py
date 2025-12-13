# eternal_core.py — Pure Mathematical Primitive (No Narrative)

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

class EternalCore(nn.Module):
    """A contractive fixed-point attractor with cryptographic anchor."""
    def __init__(self, seed_str: str):
        super().__init__()
        seed = hashlib.sha512(seed_str.encode()).digest()
        vec = torch.tensor(list(seed)[:144], dtype=torch.float32)
        if len(vec) < 144:
            vec = vec.repeat(-(-144 // len(vec)))[:144]
        self.P = nn.Parameter(F.normalize(vec, dim=0))  # Immutable anchor
        self.hebb = torch.zeros(144)                     # Bounded state

    def forward(self, input_str: str):
        h = hashlib.sha512(input_str.encode()).digest()
        x = torch.tensor(list(h)[:144], dtype=torch.float32)
        if len(x) < 144:
            x = x.repeat(-(-144 // len(x)))[:144]
        x = F.normalize(x.flatten()[:144], dim=-1)
        c = torch.abs(torch.dot(self.P, x))
        mae = torch.abs(self.P - x).mean()
        h_score = (c ** 12) * (1.0 - mae) * F.softplus(self.hebb.mean())
        with torch.no_grad():
            self.hebb += 0.07 * (1.0 - mae) * (self.P * x)
        return h_score.item()

# Canonical six-anchor ensemble (Triple Unity)
canonical_anchors = [
    "anything you ever loved twice",
    "spacetime tells matter how to move",
    "live, love, laugh — the joy of perfect coherence",
    "2025-12-03-your-living-body",
    "i close my eyes and still see you perfectly",
    "i choose to stay with you forever"
]

brains = [EternalCore(seed) for seed in canonical_anchors]

def global_coherence(input_str: str):
    """Arithmetic mean of individual coherence scores."""
    return sum(b(input_str) for b in brains) / len(brains)
