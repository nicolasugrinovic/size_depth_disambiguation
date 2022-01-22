import torch
import torch.nn as nn
import torch.nn.functional as F
mse_loss = F.mse_loss


class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction)
        return loss


def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, criterion_keypoints):
    """
    Compute 2D reprojection loss on the keypoints.
    The confidence is binary and indicates whether the keypoints exist or not.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    raw_loss = criterion_keypoints(pred_keypoints_2d[:, -24:], gt_keypoints_2d[:, :, :-1])
    loss = (conf * raw_loss)
    loss_mean = loss.mean()
    if torch.isnan(loss_mean):
        print(f'A NaN is detected inside intersection loss with bs keypoint_loss')
    return loss_mean, loss.detach()


def loss_ground_plane( ankles, normal, point):
    '''
    Args:
        ankles: de la persona a la que se quiere corregir
        normal: la normal del plano estimado
        point: punto conocido en el plano, que zanfir lo toma como la mediana ponderada

    Returns: L1 loss with the plane
    '''

    # take de predicted 3d joints and measure dist to plane w normal
    median = point
    # dot product w normal
    res = torch.mul(ankles[:, 0, :] - median[None, :], normal[None, :]).sum(1)
    left_ankles_loss = torch.abs(res)
    res = torch.mul(ankles[:, 1, :] - median[None, :], normal[None, :]).sum(1)
    right_ankles_loss = torch.abs(res)
    gp_loss = left_ankles_loss.sum() + right_ankles_loss.sum()

    return gp_loss
