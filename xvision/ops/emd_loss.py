import torch


def emd_loss(preds, targets, p=2, reduction='mean'):
    cdf_y = torch.cumsum(targets, dim=1)
    cdf_pred = torch.cumsum(preds, dim=1)
    cdf_diff = cdf_pred - cdf_y
    den = cdf_diff.size(-1) ** (1/p)
    loss = torch.norm(cdf_diff, p=p, dim=-1) / den
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


if __name__ == '__main__':
    a = torch.rand(128, 10)
    b = torch.rand(128, 10)

    loss = emd_loss(a, b)
    print(loss)