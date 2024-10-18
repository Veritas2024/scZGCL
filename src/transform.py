import torch

class Hete_DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.eps = 1e-7

    def __call__(self, data):
        if self.p == 0.0:
            return data

        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

