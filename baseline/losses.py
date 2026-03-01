import torch.nn.functional as F


def criterion(outputs, targets, lambda_glob=1.0, mu_cons=0.0):
    """
    outputs:
      G_hat: [B,4,8,8]
      S_hat: [B,1]
      S_agg: [B,1]
    targets:
      tile_cov: [B,8,8,4]
      global_score: [B,1]
    """
    G_tgt = targets["tile_cov"].permute(0, 3, 1, 2).contiguous()  # [B,4,8,8]
    loss_tile = F.l1_loss(outputs["G_hat"], G_tgt)

    loss_glob = F.l1_loss(outputs["S_hat"], targets["global_score"])

    loss_cons = F.l1_loss(outputs["S_hat"], outputs["S_agg"])

    total = loss_tile + lambda_glob * loss_glob + mu_cons * loss_cons
    logs = {
        "l_tile": float(loss_tile.detach().cpu()),
        "l_glob": float(loss_glob.detach().cpu()),
        "l_cons": float(loss_cons.detach().cpu()),
    }
    return total, logs
