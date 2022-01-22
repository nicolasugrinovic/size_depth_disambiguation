import numpy as np
import trimesh
import pyransac3d as pyrsc
import open3d as o3d
pcd = o3d.geometry.PointCloud()
from .misc import save_points
from .depth import read_joints, project_joints_to_img, vis_proj_joints_t
import torch
import math
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




def inspect_3d_joints(image, folder, name_p0, img_shape, gt_keypoints=None):

    dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()
    _, joints_p0, _ = read_joints(folder, name_p0)
    joints_p0_t = torch.tensor(joints_p0)[None].float().cuda()
    joints_projected = project_joints_to_img(joints_p0_t, img_shape, dummy_trans)
    vis_proj_joints_t(image, joints_projected, gt_keypoints)



def eval_plane(a,b,c,d,  xrange, yrange, return_pend=False):

    x0 = xrange[0]
    x1 = xrange[1]
    y0 = yrange[0]
    y1 = yrange[1]
    xx = np.linspace(x0, x1, num=50)
    yy = np.linspace(y0, y1, num=50)
    X, Y = np.meshgrid(xx, yy)
    coords_2d = np.stack([X, Y], 2).reshape([-1, 2])

    A = -1 * a / c
    B = -1 * b / c
    C = -1 * d / c
    z = A * coords_2d[:, 0] + B * coords_2d[:, 1] + C
    ground_est = np.concatenate([coords_2d, z[:, None]], 1)
    # dy = yy[0] - yy[1]
    # dz = z[0] - z[1]
    dy = Y[0, 0] - Y[1, 0]
    dz = z[0] - z[50]
    pend = -1 * dy / dz
    # pend = dz / dy

    angle = math.atan(pend)
    angle_d = math.degrees(angle)

    if return_pend:
        return ground_est, pend, angle_d
    else:
        return ground_est

def eval_save_normal_vector(a,b,c, point0):
    x = np.linspace(0, 0.1, num=50)
    x0, y0, z0 = point0
    y = (b * (x - x0) / a) + y0
    z = (c * (x - x0) / a) + z0
    norm_vect_points = np.stack([x, y, z], 1)
    out_name = './norm_vect.ply'
    save_points(norm_vect_points, out_name)

def eval_normal_vector(a,b,c,d,  xrange, yrange):

    x0 = xrange[0]
    x1 = xrange[1]
    y0 = yrange[0]
    y1 = yrange[1]
    xx = np.linspace(x0, x1, num=50)
    yy = np.linspace(y0, y1, num=50)
    X, Y = np.meshgrid(xx, yy)
    coords_2d = np.stack([X, Y], 2).reshape([-1, 2])

    A = -1 * a / c
    B = -1 * b / c
    C = -1 * d / c
    z = A * coords_2d[:, 0] + B * coords_2d[:, 1] + C
    ground_est = np.concatenate([coords_2d, z[:, None]], 1)
    return ground_est


def estimate_plane_xy_diff_range(q_3d, xrange, yrange, name='plane.ply', return_normal=False,
                                 debug=False):
    # estimate plane points
    # range mins,maxs
    # bounds = -5.0
    # xy = np.arange(-bounds, bounds, 0.1)
    # l = len(xy)
    # xy_coords = np.repeat(xy, [l]).reshape([l, l])
    # coords_2d = np.stack([xy_coords, xy_coords.T], 2).reshape([-1, 2])

    plane1 = pyrsc.Plane()
    # Results in the plane equation Ax+By+Cz+D
    blockPrint()
    best_eq, best_inliers = plane1.fit(q_3d, 0.01)
    enablePrint()
    # in the form of a'x+b'y+c'z+d'=0. --> z = (-a'x- b'y -d') / c'
    a = best_eq[0]
    b = best_eq[1]
    c = best_eq[2]
    d = best_eq[3]

    normal_vect = np.array([a, b, c])

    if debug:
        x0 = xrange[0]
        x1 = xrange[1]
        # xran = x1-x0
        y0 = yrange[0]
        y1 = yrange[1]
        # yran = y1-y0

        resolution = 100
        xx = np.linspace(x0, x1, num=resolution)
        yy = np.linspace(y0, y1, num=resolution)
        X, Y = np.meshgrid(xx, yy)
        coords_2d = np.stack([X, Y], 2).reshape([-1, 2])

        A = -1 * a / c
        B = -1 * b / c
        C = -1 * d / c
        z = A * coords_2d[:, 0] + B * coords_2d[:, 1] + C
        ground_est = np.concatenate([coords_2d, z[:, None]], 1)

        # ones = np.ones_like(z[:, None] )
        # xyin = np.concatenate([coords_2d, ones], 1)
        # save_points(xyin, './results/xy_points.ply')

        pc = trimesh.PointCloud(ground_est)
        pcd.points = o3d.utility.Vector3dVector(pc)
        # fname = "./results/estimated_ground_%s.ply"%(name)
        o3d.io.write_point_cloud(name, pcd)
        # print('plane saved at: %s' % name)

    if return_normal:
        return normal_vect
