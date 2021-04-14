import torch


def emd_loss(preds, targets):
    cdf_y = torch.cumsum(targets, dim=1)
    cdf_pred = torch.cumsum(preds, dim=1)
    cdf_diff = cdf_pred - cdf_y
    emd_loss = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
    return emd_loss.mean()

