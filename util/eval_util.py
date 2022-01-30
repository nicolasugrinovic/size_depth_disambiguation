import numpy as np
from util.misc import vectorize_distance
from scipy.optimize import linear_sum_assignment
from util.bbox import bbox_from_joints_several, box_iou_np, mask_joints_w_vis

def collate_fn(batch):
    batch_tuple = batch[0]
    return batch_tuple

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
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


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


def get_dist_matix(gt_trans_):
    all_depts = gt_trans_[:, 2]
    z_diffs = all_depts[:, None] - all_depts[None, :]
    return z_diffs

def get_heigths(j3d_height):
    jts0 = j3d_height[:, :-1, :]
    jts1 = j3d_height[:, 1:, :]
    p_diff = jts0 - jts1
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum(2)
    norm = np.sqrt(p_sqr_sum)
    heights = norm.sum(1)
    return heights

def detect_occlusions(img, j2d_gt_sc, j2d_initial, debug_this=False):
    occ_threshold = 0.5
    vis = j2d_gt_sc[0, :, 2].astype(bool)
    # vis = vis.squeeze().squeeze().astype(bool)
    j2d_gt_masked = j2d_gt_sc[:, vis]
    bbox_gt = bbox_from_joints_several(j2d_gt_masked[..., :2])
    bbox_out = bbox_from_joints_several(j2d_initial[..., :2])
    # iou is gt rows vs. out cols
    iou, _ = box_iou_np(bbox_gt, bbox_out)
    non_zero = iou > occ_threshold
    count_cols = non_zero.sum(0)
    occ = count_cols > 1
    occ_idxs = occ[None] * non_zero
    detected_occ = occ.sum()
    return detected_occ, occ_idxs

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
    return trans, j2d, scale, out_dict