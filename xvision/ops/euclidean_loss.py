import torch


def euclidean_loss(inputs, targets, reduction='none'):
    # inputs: [B, p, 2]
    # target: [B, p, 2]
    diff = inputs - targets
    loss = torch.norm(diff, p=2, dim=-1)

    if reduction == 'sum':
        loss = loss.sum()
    
    elif reduction == 'mean':
        loss = loss.mean()

    return loss


if __name__ == '__main__':
    a = torch.rand(128, 98, 2)
    b = torch.rand(128, 98, 2)

    loss = euclidean_loss(a, b, 'none')
    print(loss)
    print(loss.shape)