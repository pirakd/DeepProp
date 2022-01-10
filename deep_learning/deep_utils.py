import torch
def MaskedMean(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim) / mask.sum(dim)

def MaskedSum(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim)
