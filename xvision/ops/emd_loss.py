import torch


def emd_loss(y, y_pred):
    cdf_y = torch.cumsum(y, dim=1)
    cdf_pred = torch.cumsum(y_pred, dim=1)
    cdf_diff = cdf_pred - cdf_y
    emd_loss = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
    return emd_loss.mean()
