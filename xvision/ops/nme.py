import torch
import torch.nn as nn

# normalized mean error


def nme(inputs, targets, left, right, reduction='none'):
    # inputs: [B, p, 2]
    # targets: [B, p, 2]
    # left: left eye indice
    # right: right eye indice

    diff = inputs - targets
    me = torch.norm(diff, p=2, dim=-1).mean(dim=-1)  # [B]

    leye = targets[..., left, :]   # [B, m, 2]
    reye = targets[..., right, :]  # [B, n, 2]
    mleye = leye.mean(dim=-2)      # [B, 2]
    mreye = reye.mean(dim=-2)      # [B, 2]

    d = torch.norm(mleye - mreye, p=2, dim=-1)
    loss = me / d
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()

    return loss


class IbugScore(nn.Module):
    def __init__(self, left_eye, right_eye):
        super().__init__()
        self.left_eye = left_eye
        self.right_eye = right_eye

    def forward(self, preds, targets):
        return nme(preds, targets, self.left_eye, self.right_eye, 'mean') * 100


if __name__ == '__main__':

    inputs = torch.rand(128, 68, 2)
    targets = torch.rand(128, 68, 2)

    loss = nme(inputs, targets, [])
