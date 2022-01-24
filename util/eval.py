from util.misc import save_points, vectorize_distance
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_dz_ordered(out_trans, idxs_out):
    """ ordenado por indx pero no en orden de z"""
    out_zs = out_trans[idxs_out, 2]
    nearz = min(out_zs)
    dzs = out_zs - nearz
    return dzs


def get_data_from_dict(dict, with3D=False):
    j2d = dict['joints2d']
    scale = dict['scale']
    trans = dict['translations']
    if with3D:
        j3d = dict['joints3d']
        return j2d, trans, scale, j3d
    return j2d, trans, scale

def order_idx_by_gt_j2d(vis, j2d_gt_sc, j2d_output):
    # detect occlusion
    dist = vectorize_distance(vis * j2d_output, j2d_gt_sc[..., :2])
    idxs_out, idxs_gt = linear_sum_assignment(dist)
    return idxs_out, idxs_gt


def get_sign_matix(gt_trans_):
    all_depts = gt_trans_[:, 2]
    z_diffs = all_depts[:, None] - all_depts[None, :]
    sign_matrix = z_diffs / abs(z_diffs + 0.000000001)
    return np.ceil(sign_matrix)


def get_dist_matix(gt_trans_):
    all_depts = gt_trans_[:, 2]
    z_diffs = all_depts[:, None] - all_depts[None, :]
    return z_diffs

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def read_list(list_path):
    with open(list_path, 'r') as f:
        instances = f.read().split('\n')
        img_names = instances[:-1]
    return img_names

def get_ranks(j2d_gt_sc, j2d_initial, j3d_gt, j3d_tgt, vis):
    n_j2d_gt = len(j2d_gt_sc)
    n_j2d_init = len(j2d_initial)
    n_j3d = len(j3d_tgt)

    j3d_root_gt = j3d_gt[..., :3] - j3d_gt[:, 14, None, :3]
    j3d_root_tgt = j3d_tgt[..., :3] - j3d_tgt[:, 14, None, :3]
    j3d_root_tgt = vis * j3d_root_tgt
    dist = vectorize_distance(j3d_root_gt, j3d_root_tgt)
    rank = np.linalg.matrix_rank(dist)
    max_r = max(n_j2d_gt, n_j2d_init)
    return rank, max_r,  n_j2d_gt, n_j3d


def order_idx_by_gt_j2d(vis, j2d_gt_sc, j2d_output):
    # detect occlusion
    dist = vectorize_distance(vis * j2d_output, j2d_gt_sc[..., :2])
    idxs_out, idxs_gt = linear_sum_assignment(dist)
    return idxs_out, idxs_gt


def get_heigths(j3d_height):
    jts0 = j3d_height[:, :-1, :]
    jts1 = j3d_height[:, 1:, :]
    p_diff = jts0 - jts1
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum(2)
    norm = np.sqrt(p_sqr_sum)
    heights = norm.sum(1)
    return heights


def parse_data(data_dict, with3d=False):
    trans = data_dict['translations']
    j2d = data_dict['joints2d']
    scale = data_dict['scale']
    trans = np.array(trans)
    j2d = np.array(j2d)
    scale = np.array(scale)

    out_dict = {
        'scale': scale.squeeze(),
        'translations': trans,
        'joints2d': j2d,
    }
    if with3d:
        j3d = data_dict['joints3d']
        j3d = np.array(j3d)
        add_dict = {
            'joints3d': j3d,

        }
        out_dict.update(add_dict)
    return trans, j2d, scale, j3d, out_dict



"""
Utils for evaluation.
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np


def compute_similarity_transform(S1, S2, return_transf=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    if return_transf:
        return S1_hat, t, scale
    else:
        return S1_hat


def align_by_pelvis(joints, get_pelvis=False):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa


def compute_mpjpe_errors(gt3d, pred, return_transf=False):
    """
    Modificado del e arriba por mi nuk
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    gt3d = gt3d.reshape(-1, 3)
    pred = pred.reshape(-1, 3)
    # Root align.
    # gt3d = align_by_pelvis(gt3d)
    # pred3d = align_by_pelvis(pred)

    joint_error = np.sqrt(np.sum((gt3d - pred)**2, axis=1))
    errors = np.mean(joint_error)

    # Get PA error.
    if return_transf:
        pred3d_sym, t, scale = compute_similarity_transform(pred, gt3d, return_transf=True)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa = np.mean(pa_error)
        return errors, errors_pa, t, scale, pred3d_sym
    else:
        pred3d_sym = compute_similarity_transform(pred, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa = np.mean(pa_error)
        return errors, errors_pa
