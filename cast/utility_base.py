"""Shared utility components for the public CAST package."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Utility(nn.Module):
    def __init__(self, dim, mode='full'):
        super().__init__()
        self.mode = mode
        self.p = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.adapter = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gamma = nn.Parameter(torch.tensor(-3.0))
        self.norm = nn.LayerNorm(dim)

    def task_adapter(self, task):
        d = task.shape[-1]
        attention = (self.q(task) @ self.k(self.p).T / math.sqrt(d)).softmax(-1)
        preference = attention @ self.p
        return F.normalize(self.norm(task + torch.sigmoid(self.gamma) * self.adapter(
            torch.cat([task, preference, task * preference], -1))), dim=-1)
