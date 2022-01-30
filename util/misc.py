import numpy as np
from PIL import Image
import cv2
import trimesh
import pickle
import torch
import json
import os
import open3d as o3d
import scipy.io
import scipy.io as scio


pcd = o3d.geometry.PointCloud()
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def get_heigths(j3d_height):
    jts0 = j3d_height[:, :-1, :]
    jts1 = j3d_height[:, 1:, :]
    p_diff = jts0 - jts1
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum(2)
    norm = np.sqrt(p_sqr_sum)
    heights = norm.sum(1)
    return heights


def get_distance(points):
    p1, p2 = points
    p_diff = p1 - p2
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum()
    norm = np.sqrt(p_sqr_sum)
    return norm


def draw_lsp_14kp__bone(img_pil, pts):
    bones = [
        [0, 1, 255, 0, 0],
        [1, 2, 255, 0, 0],
        [2, 12, 255, 0, 0],
        [3, 12, 0, 0, 255],
        [3, 4, 0, 0, 255],
        [4, 5, 0, 0, 255],
        [12, 9, 0, 0, 255],
        [9, 10, 0, 0, 255],
        [10, 11, 0, 0, 255],
        [12, 8, 255, 0, 0],
        [8, 7, 255, 0, 0],
        [7, 6, 255, 0, 0],
        [12, 13, 0, 255, 0]
    ]
    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    for pt in pts:
        if pt[2] > 0.2:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        if pa[2] > 0.2 and pb[2] > 0.2:
            cv2.line(img, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 2)

    # plot(img)
    return img



def vectorize_distance(a, b):
    """
    Calculate euclidean distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape(N, -1)
    M = b.shape[0]
    b = b.reshape(M, -1)
    a2 = np.tile(np.sum(a ** 2, axis=1).reshape(-1, 1), (1, M))
    b2 = np.tile(np.sum(b ** 2, axis=1), (N, 1))
    dist = a2 + b2 - 2 * (a @ b.T)
    dist[np.where(dist<-0.0)] = 0
    return np.sqrt(dist)

def save_points(rxz, name='points.ply'):
    pc = trimesh.PointCloud(rxz)
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(name, pcd)

def mask_joints_w_vis(j2d):
    vis = j2d[0, :, 2].astype(bool)
    j2d_masked = j2d[:, vis]
    return j2d_masked

def joints_delete_zeros(j2d):
    vis = (j2d.sum(2) > 0).astype(bool)[0]
    j2d_masked = j2d[:, vis]
    return j2d_masked

def joints_delete_zeros_v1(j2d):
    vis = (j2d.sum(2) > 0).astype(bool)
    vis = np.all(vis, axis=0)
    j2d_masked = j2d[:, vis]
    return j2d_masked

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def read_pickle(f):
    with open(f, 'rb') as data:
        x = pickle.load(data)
    return x

def read_pickle_compatible(f):
    with open(f, 'rb') as data:
        u = pickle._Unpickler(data)
        u.encoding = 'latin1'
        p = u.load()
    return p


def read_list(list_path):
    with open(list_path, 'r') as f:
        instances = f.read().split('\n')
        img_names = instances[:-1]
    return img_names


def write_pickle(f, data):
    with open(f, 'wb') as f:
        pickle.dump(data, f)

def read_mat(f):
    mat = scipy.io.loadmat(f)
    return mat

def write_mat(f, data):
    scio.savemat(f, data)



def plot(img):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img)
    plt.show()


def plot_boxes_cv2(img_pil, boxes, do_plot=True):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)


    for (xmin, ymin, xmax, ymax), c in zip(boxes.tolist(), COLORS * 100):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)


    if do_plot:
        plot(img)
    else:
        return img


def plot_boxes_w_text_cv2(img_pil, boxes, number_list, do_plot=True, fontScale=2):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    for (xmin, ymin, xmax, ymax), c, number in zip(boxes.tolist(), COLORS * 100, number_list):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        ymid = int((ymin + ymax) / 2)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        text = '%.2f' % number
        img = cv2.putText(img, text, (xmin, ymin+50), font, fontScale=fontScale, color=(255, 0, 0), thickness=2)

    if do_plot:
        plot(img)
    else:
        return img

def plot_boxes_w_persID_cv2(img_pil, boxes, number_list, do_return=False, fontScale=4):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    for (xmin, ymin, xmax, ymax), c, number in zip(boxes.tolist(), COLORS * 100, number_list):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        ymid = int((ymin + ymax) / 2)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        text = '%d' % number
        img = cv2.putText(img, text, (xmin, ymid), font, fontScale=fontScale, color=(255, 0, 0), thickness=6)

    if do_return:
        return img
    else:
        plot(img)

def add_txt_to_img(img_pil, text, x=0,y=100, fontSize=2, thickness=4):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    # x = 0
    # y = 100

    img = cv2.rectangle(img, (x, y+20), (x + 2000, y+10-100), (0, 255, 0), -1)
    img = cv2.putText(img, text, (x, y), font, fontScale=fontSize,
                      color=(255, 0, 0), thickness=thickness)
    return img

def add_txt_to_img_w_pad(img_pil, text, x=0,y=100, fontSize=2, thickness=4):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    # x = 0
    # y = 100
    h, w, c = img.shape
    new_h = int(h * 1.1)
    zero_img = np.zeros([new_h, w, c], dtype=np.uint8)
    offs = new_h - h
    zero_img[offs:, ...] = img
    # plot(zero_img)
    img = zero_img
    img = cv2.rectangle(img, (x, y+20), (x + 2000, y+10-100), (0, 255, 0), -1)
    img = cv2.putText(img, text, (x, y), font, fontScale=fontSize,
                      color=(255, 0, 0), thickness=thickness)
    # plot(img)


    return img

def plot_joints_cv2(img_pil, gtkps, do_plot=True, with_text=False):
    '''
    :param img_pil:
    :param gtkps: should be of dims n x K x 3 or 2 -> [n,K,3], joint locations should be integer
    :param do_plot:
    :return:
    '''
    font = cv2.FONT_HERSHEY_PLAIN
    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    h, w, _ = img.shape
    max_s = max(h, w)
    sc = int(max_s / 500)
    # convert to int for cv2 compat
    if isinstance(gtkps, np.ndarray):
        gtkps = gtkps.astype(np.int)
    elif isinstance(gtkps, torch.Tensor):
        gtkps = gtkps.int().numpy()
    else:
        print('Unknown type!!')

    for kpts in gtkps:
        for i, (x, y) in enumerate(kpts[..., :2]):
            img = cv2.circle(img, (x, y), radius=2*sc, color=(255, 255, 0), thickness=2*sc)
            if with_text:
                text = '%d' % i
                img = cv2.putText(img, text, (x, y), font, fontScale=1.0*sc, color=(255, 0, 0), thickness=1*sc)

    if do_plot:
        plot(img)
    else:
        return img


def save_mesh(vertices, out_path='scene.obj'):
    smpl_model_path = './models/smpl_faces.npy'
    faces = np.load(smpl_model_path)

    tri_verts = []
    for v in vertices:
        triv = trimesh.Trimesh(v, faces)
        tri_verts.append(triv)

    scene = tri_verts[0].scene()
    for v in tri_verts[1:]:
        scene.add_geometry(v)
    scene.export(out_path)



