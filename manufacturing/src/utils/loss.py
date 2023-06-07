import torch.nn as nn
import torch


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        target_size = target.size()
        preds = preds.reshape(-1, len(self.quantiles))
        target = target.reshape(-1, 1)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i].reshape(-1, 1)
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ))
        loss = torch.sum(torch.stack(losses, dim=1), dim=1)
        return loss.reshape(target_size)
