import torch


def emd_loss(preds, targets, p=2, reduction='mean'):
    '''Impolement Earth Mover's Distance loss

    This loss function was proposed in paper `NIMA: Neural Image Assessment`_

    Args:
        preds (Tensor): batched predictions of the probability mass function.
            The shape should be [N, C]
        targets (Tensor): target probability mass function same shape with preds
        p (int, optional): order of p-norm. Defaults to 2.
        reduction (str, optional): reduction method which can be one of 'mean', 'sum' 
            or none. Defaults to 'mean'.

    Returns:
        [Tensor]: the loss
    .. _NIMA\: Neural Image Assessment: http://arxiv.org/abs/1709.05424
    '''
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
