import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss
from xvision.ops import euclidean_loss


def score_box_loss(target_scores, target_deltas, pred_logits, pred_deltas):
    # target_scores: [B, k]
    # target_deltas: [B, k, 4]
    # pred_scores: [B, k]
    # pred_deltas: [B, k, 4]
    score_loss = sigmoid_focal_loss(
        pred_logits, target_scores, reduction='sum')
    maxscore, _ = target_scores.max(-1)
    pos_mask = maxscore > 0.5

    target_box_deltas_pos = target_deltas[pos_mask].reshape(-1, 2, 2)
    pred_box_deltas_pos = pred_deltas[pos_mask].reshape(-1, 2, 2)
    box_loss = euclidean_loss(
        pred_box_deltas_pos, target_box_deltas_pos, 'sum')
    npos = target_box_deltas_pos.shape[0]

    return score_loss / npos, box_loss / npos


def score_box_point_loss(target_scores, target_box_deltas, target_point_deltas, pred_logits, pred_box_deltas, pred_point_deltas, point_mask):
    # target_scores: [B, k]
    # target_box_deltas: [B, k, 4]
    # target_point_deltas: [B, k, p, 2]
    # point_mask: [B, k]

    score_loss = sigmoid_focal_loss(
        pred_logits, target_scores, reduction='sum')
    maxscore, _ = target_scores.max(-1)
    pos_mask = maxscore > 0.5

    target_box_deltas_pos = target_box_deltas[pos_mask].reshape(-1, 2, 2)
    pred_box_deltas_pos = pred_box_deltas[pos_mask].reshape(-1, 2, 2)
    box_loss = euclidean_loss(
        pred_box_deltas_pos, target_box_deltas_pos, 'sum')

    point_mask = pos_mask & point_mask

    target_point = target_point_deltas[point_mask]  # -1, p, 2
    pred_point = pred_point_deltas[point_mask]      # -1, p, 2

    point_loss = euclidean_loss(pred_point, target_point, 'sum')

    npos = target_box_deltas_pos.shape[0]
    npoint = target_point.shape[0]

    return score_loss / npos, box_loss / npos, point_loss / npoint
