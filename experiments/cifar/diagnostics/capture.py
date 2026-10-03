import torch
from torchdeq.utils.layer_utils import MDEQWrapper


class Captured(Exception):
    pass


class Capture(torch.nn.Module):
    def forward(self, func, initial, **kwargs):
        self.func = MDEQWrapper(func, initial)
        self.initial = self.func.list2vec(initial)
        raise Captured()
