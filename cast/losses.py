"""Directional content omission objective, separate from graph InfoNCE."""
from torch.nn import functional as F


def directional_content_loss(original,masked,margin=0.1,temperature=1.0):
    if original.shape != masked.shape or original.numel()==0:
        raise ValueError('Expected nonempty aligned utilities')
    if temperature<=0:
        raise ValueError('Temperature must be positive')
    return F.softplus((margin-(original-masked))/temperature).mean()
