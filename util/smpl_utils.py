
import torch

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def rotmat_to_axisAngle(rotmat):
    rotmat = rotmat.view(-1, 3, 3)
    tr = rotmat[:, 0 , 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    # todo aparecen puros nans aqui
    angle = torch.acos(tr - 1 / 2)
    ax = (rotmat[:, 2 , 1] - rotmat[:, 1, 2]) / (2 * torch.sin(angle))
    ay = (rotmat[:, 0 , 2] - rotmat[:, 2, 0]) / (2 * torch.sin(angle))
    az = (rotmat[:, 1 , 0] - rotmat[:, 0, 1]) / (2 * torch.sin(angle))
    axis_angle = angle * torch.tensor([ax, ay, az])
    return axis_angle



def smpl2rot_gen(thetas):
    """
    torch input
    """
    # todo make this more general
    thetas = thetas.view(-1)
    thetas = thetas.view(-1, 3)
    thetas_rot = batch_rodrigues(thetas)
    thetas_rot = thetas_rot.view(-1, 24, 3, 3)
    return thetas_rot