"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import sys
SEG_PATH = './external/panoptic_deeplab'
sys.path.insert(1, SEG_PATH)

from util import plot, get_person_depth, plot_joints_cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from util.plane import inspect_3d_joints, estimate_plane_xy_diff_range, eval_plane
from util.depth import read_ankles, read_all_joints, project_joints_to_img
from util.misc import save_points, vectorize_distance

from detectron2.data.detection_utils import read_image
from PIL import Image

from external.panoptic_deeplab.tools_d2.d2.predictor import VisualizationDemo
import matplotlib.pyplot as plt
from util import get_distance, bbox_from_joints_several

from util.detectron import setup_cfg, get_keypoints_predictor
from constants import DETECTRON17_TO_24
from util.model import get_args, init_network, get_prediction
import tqdm
from util.misc import plot_boxes_cv2, plot_boxes_w_text_cv2, joints_delete_zeros, plot_boxes_w_persID_cv2
from util.bbox import compute_iou, box_iou_np
from util.depth import mask_distance_reverse, get_sign_matix_from_depths, upper_tri_masking_torch, upper_tri_masking
import math
import pickle



debug = False
# debug = True
# debug1 = True
debug1 = False


def run_w_reproj_paper(args, input_path, output_path, model_path, model_type="large", optimize=True):
    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    model, net_w, net_h, resize_mode, normalization, transform = init_network(model_type, model_path, device, optimize)

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    # get input
    input_path = args.input_path
    img_names = glob.glob(os.path.join(input_path, "*"))
    img_names = [c for c in img_names if ('.jpg' in c or '.png' in c) and 'output.' not in c]
    img_names.sort()
    # img_names = img_names[126:]
    num_images = len(img_names)
    print('Number of images to process: %d' % num_images)
    dir_name = ''
    num_wrong_onePerson = 0
    print('Doing folder: %s' % dir_name)
    print('mode: %s' % args.mode)
    print('ordinal weight: %d' % args.w_ordinal_loss)
    plane_scale = args.plane_scale
    n_iters = args.n_iters
    print('plane escale: %.1f' % plane_scale)
    print('number of iters: %d' % n_iters)
    output_path = os.path.join(args.output_path, dir_name)
    os.makedirs(output_path, exist_ok=True)
    out_dir = output_path
    print('output folder: %s' % out_dir)

    for ind, img_name in tqdm.tqdm(enumerate(img_names), total=num_images):
        args.input_path = os.path.dirname(img_name)
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # compute

        iname = os.path.basename(img_name).split('.')[0]



        prediction = get_prediction(img, img_input, model, optimize, device)
        name = '%s_3djoints_0.json' % iname
        image = Image.open(img_name)
        w, h = image.size
        image = np.array(image)
        image = image[..., :3]

        # output
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, os.path.splitext(os.path.basename(img_name))[0])
        out_depth_16bits = utils.write_depth(filename, prediction, bits=2)
        out_depth = (1000*prediction).astype(np.float32)
        if debug:
            save_raw_scaled_disparities(out_depth, out_depth_16bits, w, h, iname, factor=10)
            plot(image)
            plot(out_depth)

        # using this output, its normalized btw [0, 65k]
        out_depth = out_depth_16bits
        zero_idxs = np.where(out_depth == 0.0)
        mean_disp = out_depth.mean()
        out_depth[zero_idxs] = mean_disp
        predicted_disparity = out_depth

        # use PIL, to be consistent with evaluation
        img = read_image(img_name, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # plot(img)
        os.makedirs('./results', exist_ok=True)
        out_filename = './results/panoptic_%d.png' % np.random.randint(0, 1000)
        visualized_output.save(out_filename)
        # pan_img = plt.imread(out_filename)
        # plot(pan_img)

        pan = predictions['panoptic_seg'][0].cpu().numpy()
        pan = (pan / 1000).astype(int)
        labels = np.unique(pan)
        classes = []



        for l in labels:
            classes.append(demo.metadata.stuff_classes[l])
        # para saber el id y el nombre de la clase de suelo ver las variables: labels, clases
        road = pan == 100
        grass = pan == 125
        rug_merged = pan == 132
        sand = pan == 102
        playingfield = pan == 97
        snow = pan == 105
        floor_other_merged = pan == 122
        dirt_merged = pan == 126
        pavement_merged = pan == 123
        floor_wood = pan == 87
        persons_mask = pan == 0

        road = road | grass | rug_merged | sand | playingfield | snow | \
               floor_other_merged | dirt_merged | pavement_merged | floor_wood
        # plot(road)

        # folder = os.path.join(args.input_path, 'annots')
        folder = args.input_path
        joint_files = os.listdir(folder)
        joint_names = [c for c in joint_files if iname in c and '.json' in c]
        if len(joint_names)==0:
            print('No joints .json!!!')
            continue

        joint_names.sort()
        n_people_crmp = len(joint_names)
        img_shape = torch.Tensor([832, 512])[None]
        dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()
        # read all joints
        _, joints_trans_all, trans_all = read_all_joints(folder, joint_names)
        joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
        all_joints_projected = project_joints_to_img(joints_trans_all_t, img_shape, dummy_trans)
        glob_kpts_all = torch.ones([n_people_crmp, 24, 3], device=dummy_trans.device)
        glob_kpts_all[..., :2] = all_joints_projected
        # kpts = glob_kpts_all

        if n_people_crmp == 1:
            num_wrong_onePerson += 1
            print('only one person, '
                  'file={}, num_wrong_onePerson={}'.format(filename, num_wrong_onePerson))
            continue

        joint_names.sort()
        names_ord = [int(c.split('3djoints_')[-1].split('.')[0]) for c in joint_names]
        names_ord = np.array(names_ord)
        sort_idxs = np.argsort(names_ord)
        joint_names = np.array(joint_names)
        # real sorted including 2 digit strings
        joint_names = joint_names[sort_idxs]
        iname = name.replace('_3djoints_0.json', '')

        result_dir = os.path.join(out_dir, iname)
        os.makedirs(result_dir, exist_ok=True)

        data = {
            'predicted_disparity': predicted_disparity,
            'road': road,
            'h': h,
            'w': w,
            'glob_kpts_all': glob_kpts_all,  #these are reprojected from the estimations directly
            'joint_names': joint_names,
            'persons_mask': persons_mask,
        }
        # so with this data, reprojection wont happen with Detectron output but from initial est.
        out_file = os.path.join(result_dir, f'data_reproj_joints.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)

        img_out = os.path.join(result_dir, 'image.jpg')
        plt.imsave(img_out, image)
        img_out = os.path.join(result_dir, 'depth_map.jpg')
        out_depth_3 = np.stack([out_depth, out_depth, out_depth], 2) / out_depth.max()
        plt.imsave(img_out, out_depth_3)
        img_out = os.path.join(result_dir, 'depth_ground.jpg')
        plt.imsave(img_out, out_depth_3 * road[..., None])



    print("finished")


def run_w_det_paper(args, input_path, output_path, model_path, model_type="large", optimize=True):
    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    model, net_w, net_h, resize_mode, normalization, transform = init_network(model_type, model_path, device, optimize)

    kpts_predictor = get_keypoints_predictor()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    # get input
    input_path = args.input_path
    img_names = glob.glob(os.path.join(input_path, "*"))
    img_names = [c for c in img_names if ('.jpg' in c or '.png' in c) and 'output.' not in c]
    img_names.sort()
    # img_names = img_names[126:]
    num_images = len(img_names)
    print('imgs todo: %d' % num_images)
    dir_name = ''
    num_wrong_onePerson = 0
    print('Doing folder: %s' % dir_name)
    print('mode: %s' % args.mode)
    print('ordinal weight: %d' % args.w_ordinal_loss)
    plane_scale = args.plane_scale
    n_iters = args.n_iters
    print('plane escale: %.1f' % plane_scale)
    print('number of iters: %d' % n_iters)
    output_path = os.path.join(args.output_path, dir_name)
    os.makedirs(output_path, exist_ok=True)
    out_dir = output_path
    print('output folder: %s' % out_dir)

    for ind, img_name in tqdm.tqdm(enumerate(img_names), total=num_images):
        args.input_path = os.path.dirname(img_name)
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # compute
        prediction = get_prediction(img, img_input, model, optimize, device)

        iname = os.path.basename(img_name).split('.')[0]
        name = '%s_3djoints_0.json' % iname
        image = Image.open(img_name)
        w, h = image.size
        image = np.array(image)
        image = image[..., :3]

        # output
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, os.path.splitext(os.path.basename(img_name))[0])
        out_depth_16bits = utils.write_depth(filename, prediction, bits=2)
        out_depth = (1000*prediction).astype(np.float32)
        if debug:
            save_raw_scaled_disparities(out_depth, out_depth_16bits, w, h, iname, factor=10)
            plot(image)
            plot(out_depth)

        # using this output, its normalized btw [0, 65k]
        out_depth = out_depth_16bits
        zero_idxs = np.where(out_depth == 0.0)
        mean_disp = out_depth.mean()
        out_depth[zero_idxs] = mean_disp
        predicted_disparity = out_depth

        # use PIL, to be consistent with evaluation
        img = read_image(img_name, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # plot(img)

        out_filename = './results/panoptic_%d.png' % np.random.randint(0, 1000)
        visualized_output.save(out_filename)
        # pan_img = plt.imread(out_filename)
        # plot(pan_img)

        pan = predictions['panoptic_seg'][0].cpu().numpy()
        pan = (pan / 1000).astype(int)
        labels = np.unique(pan)
        classes = []

        for l in labels:
            classes.append(demo.metadata.stuff_classes[l])
        # para saber el id y el nombre de la clase de suelo ver las variables: labels, clases
        road = pan == 100
        grass = pan == 125
        rug_merged = pan == 132
        sand = pan == 102
        playingfield = pan == 97
        snow = pan == 105
        floor_other_merged = pan == 122
        dirt_merged = pan == 126
        pavement_merged = pan == 123
        floor_wood = pan == 87

        persons_mask = pan == 0
        persons_n = persons_mask.sum()
        # plot(persons_mask)

        road_n = road.sum()
        grass_n = grass.sum()
        rug_n = rug_merged.sum()
        sand_n = sand.sum()
        playingfield_n = playingfield.sum()
        snow_n = snow.sum()
        floor_other_n = floor_other_merged.sum()
        dirt_merged_n = dirt_merged.sum()
        pavement_merged_n = pavement_merged.sum()
        floor_wood_n = floor_wood.sum()

        road = road | grass | rug_merged | sand | playingfield | snow | \
               floor_other_merged | dirt_merged | pavement_merged | floor_wood
        # plot(road)

        # folder = os.path.join(args.input_path, 'annots')
        folder = args.input_path
        joint_files = os.listdir(folder)
        joint_names = [c for c in joint_files if iname in c and '.json' in c]
        if len(joint_names)==0:
            print('No joints .json!!!')
            continue
        joint_names.sort()

        n_people_crmp = len(joint_names)

        img_shape = torch.Tensor([832, 512])[None]
        dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()
        # read all joints
        _, joints_trans_all, trans_all = read_all_joints(folder, joint_names)
        joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
        all_joints_projected = project_joints_to_img(joints_trans_all_t, img_shape, dummy_trans)
        glob_kpts_all = torch.ones([n_people_crmp, 24, 3], device=dummy_trans.device)
        glob_kpts_all[..., :2] = all_joints_projected
        # kpts = glob_kpts_all

        outputs = kpts_predictor(image)
        kpts = outputs["instances"].pred_keypoints
        n, k, d = kpts.shape
        det_kpts_glob = torch.zeros([n, 24, d], device=kpts.device)
        det_kpts_glob[:, DETECTRON17_TO_24] = kpts
        # plot_joints_cv2(img, det_kpts_glob[..., :2].cpu())

        # if person in kpts is too small ignore
        # bboxes = bbox_from_joints_several(kpts[..., :2].cpu().numpy())
        # areas = compute_area_several(bboxes)
        # too_small = areas > 2000
        # if too small take it out

        if n_people_crmp == 1:
            num_wrong_onePerson += 1
            print('only one person, '
                  'file={}, num_wrong_onePerson={}'.format(filename, num_wrong_onePerson))
            continue


        j2d_gt_sc_masked = joints_delete_zeros(det_kpts_glob.cpu().numpy())
        bboxes_gt = bbox_from_joints_several(j2d_gt_sc_masked[..., :2])
        bboxes = bbox_from_joints_several(glob_kpts_all[..., :2].cpu().numpy())
        # plot_boxes_w_persID_cv2(img, bboxes_gt, np.arange(0, len(bboxes_gt)), do_return=False)
        # plot_boxes_w_persID_cv2(img, bboxes, np.arange(0, len(bboxes)), do_return=False)

        # wih GT delete impossible canditates, people not in the main focus of dataset
        # do this before ordering
        iou_boxes, _ = box_iou_np(bboxes, bboxes_gt)
        iou_boxes_thr = iou_boxes > 0.2
        iou_boxes_mask = iou_boxes_thr.sum(1).astype(bool)
        # delete trans and j2d with this mask
        j2d_initial_ = glob_kpts_all[iou_boxes_mask]
        bboxes_ = bboxes[iou_boxes_mask]
        # no solo es necesario eliminar candidates with masks but also use iou in the assignation
        # cause if not, then a bad assignation may be given by the occlusion cases
        # eliminate the rest of candidates and then order?
        iou_boxes_rest, _ = box_iou_np(bboxes_, bboxes_gt)
        gt_id2assign = np.argmax(iou_boxes_rest, 1)
        j2d_gt_sc_ = det_kpts_glob[gt_id2assign]
        # vis = j2d_gt_sc_[0, :, 2][None, :, None]
        # idxs_initial, idxs_gt = order_idx_by_gt_j2d(vis.cpu().numpy(), j2d_gt_sc_.cpu().numpy()[...,:2], j2d_initial_.cpu().numpy()[..., :2])

        if debug:
            bboxes_gt_ = bboxes_gt[gt_id2assign]
            bboxes_gt_ord = bboxes_gt_
            bboxes_ord = bboxes_
            # bboxes_gt_ord = bboxes_gt_[idxs_gt]
            # bboxes_ord = bboxes_[idxs_initial]
            plot_boxes_w_persID_cv2(img, bboxes_gt_ord, np.arange(0, len(bboxes_gt_ord)), do_return=False)
            plot_boxes_w_persID_cv2(img, bboxes_ord, np.arange(0, len(bboxes_ord)), do_return=False)

        det_kpts_glob_sorted = det_kpts_glob[gt_id2assign]

        joint_names.sort()
        names_ord = [int(c.split('3djoints_')[-1].split('.')[0]) for c in joint_names]
        names_ord = np.array(names_ord)
        sort_idxs = np.argsort(names_ord)
        joint_names = np.array(joint_names)
        # real sorted including 2 digit strings
        joint_names = joint_names[sort_idxs]
        iname = name.replace('_3djoints_0.json', '')

        result_dir = os.path.join(out_dir, iname)
        os.makedirs(result_dir, exist_ok=True)

        data = {
            'predicted_disparity': predicted_disparity,
            'road': road,
            'h': h,
            'w': w,
            'glob_kpts_all': glob_kpts_all,  #these are reprojected from the estimations directly
            'det_kpts_glob_sorted': det_kpts_glob_sorted,
            'joint_names': joint_names,
            'persons_mask': persons_mask,
        }
        # so with this data, reprojection wont happen with Detectron output but from initial est.
        out_file = os.path.join(result_dir, f'data_reproj_joints.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)
        # with open(out_file, 'rb') as f:
        #     data = pickle.load(f)

        img_out = os.path.join(result_dir, 'image.jpg')
        plt.imsave(img_out, image)
        img_out = os.path.join(result_dir, 'depth_map.jpg')
        out_depth_3 = np.stack([out_depth, out_depth, out_depth], 2) / out_depth.max()
        plt.imsave(img_out, out_depth_3)
        img_out = os.path.join(result_dir, 'depth_ground.jpg')
        plt.imsave(img_out, out_depth_3 * road[..., None])

        pass

    print("finished")


def order_masks_depths_by_kpts(image, masks_person, depths, kpts, w, h):
    kpts_int = kpts.int().cpu().numpy()
    masks_arr = masks_person.cpu().numpy()

    n_people_detectron = len(kpts)
    n_masks = len(masks_arr)
    if n_masks != n_people_detectron:
        return None

    # aqui ordeno los joints para coincidir con masks
    cost = mask_distance_reverse(masks_arr, kpts_int, w, h)
    # esta funcion minimiza el costo! no al reves ojo
    idxs_masks, idxs_joints = linear_sum_assignment(cost.T)
    masks_arr_ord = masks_arr[idxs_masks]
    depths_ord = depths[idxs_masks]
    kpts_ord = kpts[idxs_joints]
    if debug:
        # plot_joints_cv2(image, kpts_int)
        plot(masks_arr[0])
        plot(masks_arr[1])
        plot(masks_arr[2])
        # plot(masks_arr[3])
        # plot(masks_arr[4])

        plot_joints_cv2(image, kpts_int[0, None])
        plot_joints_cv2(image, kpts_int[1, None])
        plot_joints_cv2(image, kpts_int[2, None])

        kpts_int_ord = kpts_int[idxs_joints]
        plot_joints_cv2(image, kpts_int_ord[0, None])
        plot(masks_arr_ord[0])
        plot_joints_cv2(image, kpts_int_ord[1, None])
        plot(masks_arr_ord[1])
        plot_joints_cv2(image, kpts_int_ord[2, None])
        plot(masks_arr_ord[2])

    return masks_arr_ord, depths_ord, kpts_ord

def get_depths_from_instance(instances, out_depth):
    is_person = instances.pred_classes == 0
    bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
    if bboxes_person.size == 0:
        return None
    masks_person = instances[is_person].pred_masks
    mean_disp = []
    union_mask = torch.zeros_like(masks_person[0], dtype=bool)
    for i, mask in enumerate(masks_person):
        union_mask = masks_person[i] + union_mask
        mean_, disp_masked = get_person_depth(out_depth, masks_person, i)
        mean_disp.append(mean_)

    if debug:
        plot(union_mask.cpu())

    mean_disp_ = torch.stack(mean_disp, 0) / 65535
    mean_disp_ = 10 * (1 - mean_disp_)
    depths = mean_disp_.cpu().numpy()
    return depths, masks_person, bboxes_person, union_mask

def get_depths_from_panoptic(predictions, out_depth):
    instances = predictions['instances']
    is_person = instances.pred_classes == 0
    bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()
    if bboxes_person.size == 0:
        return None
    masks_person = instances[is_person].pred_masks
    mean_disp = []
    union_mask = torch.zeros_like(masks_person[0], dtype=bool)
    for i, mask in enumerate(masks_person):
        union_mask = masks_person[i] + union_mask
        mean_, disp_masked = get_person_depth(out_depth, masks_person, i)
        mean_disp.append(mean_)

    if debug:
        plot(union_mask.cpu())

    mean_disp_ = torch.stack(mean_disp, 0) / 65535
    mean_disp_ = 10 * (1 - mean_disp_)
    depths = mean_disp_.cpu().numpy()
    return depths, masks_person, bboxes_person, union_mask

def get_img_coords_flatten(w, h):
    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    x_coords = w_range.T
    y_coords = np.repeat(h_range, [w]).reshape([h, w])
    assert len(x_coords) == len(y_coords)
    x_coords_f = x_coords.reshape([-1, 1])
    y_coords_f = y_coords.reshape([-1, 1])
    return x_coords_f, y_coords_f

def save_raw_scaled_disparities(out_depth, out_depth_16bits, w, h, iname, factor=100):
    idisp_16bits = -1 * out_depth_16bits / factor
    idisp = -1 * out_depth / factor
    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    x_coords = w_range.T
    y_coords = np.repeat(h_range, [w]).reshape([h, w])
    assert len(x_coords) == len(y_coords)
    x_coords_f = x_coords.reshape([-1, 1])
    y_coords_f = y_coords.reshape([-1, 1])
    idisp_16bits_flat = idisp_16bits.reshape([-1, 1])
    idisp_flat = idisp.reshape([-1, 1])
    q_3d = np.concatenate([x_coords_f, y_coords_f, idisp_16bits_flat], 1)
    save_points(q_3d, './inspect/%s_disp_16bits.ply' % iname)
    q_3d = np.concatenate([x_coords_f, y_coords_f, idisp_flat], 1)
    out_file = './inspect/%s_disp.ply' % iname
    save_points(q_3d, out_file)
    print(out_file)




def check_hip_depths(image, folder, predicted_disparity, all_gt_keypoints, joint_names):

    img_shape = torch.Tensor([832, 512])[None]
    dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()

    # read all joints
    _, joints_trans_all, _ = read_all_joints(folder, joint_names)
    joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
    all_joints_projected = project_joints_to_img(joints_trans_all_t, img_shape, dummy_trans)
    plot_joints_cv2(image, all_joints_projected.cpu().numpy())

    glb_vis = (all_gt_keypoints[..., -1] > 0.0).float()[0].cpu().numpy()
    all_joints_projected = all_joints_projected.cpu().numpy()
    is_missing_det_kpt = len(all_joints_projected) > len(all_gt_keypoints)
    dist = vectorize_distance(glb_vis[None, :, None] * all_joints_projected[..., :2],
                              all_gt_keypoints[..., :2].cpu().numpy()
                              )
    # first is for pred_kp_scl, second for glob_kpts
    idxs_pred, idxs_glob = linear_sum_assignment(dist)
    sorted_smpl_joints = all_joints_projected[idxs_pred]
    sorted_gt_kpts = all_gt_keypoints[idxs_glob].cpu().numpy()

    plot_joints_cv2(image, sorted_gt_kpts)
    # get hips
    # el detector de joints detecta mas personas en algunos casos esto ayuda a filtrar
    all_hips = sorted_gt_kpts[:, [2, 3]]
    plot_joints_cv2(image, all_hips)
    plot_joints_cv2(predicted_disparity, all_hips)

    # depths (disparities) at hips
    xy_hips = all_hips[..., :2].astype(np.int)
    # plot_joints_cv2(image, projected_hips, with_text=True)
    # plot_joints_cv2(predicted_disparity, projected_hips, with_text=True)

    # perspective projected ankles, to get the img coords and take the disp values
    idx_pers = 1
    xp1, yp1 = xy_hips[idx_pers, 0]
    xp2, yp2 = xy_hips[idx_pers, 1]
    plot_joints_cv2(predicted_disparity, xy_hips[idx_pers, [0, 1]][None], with_text=True)

    # disparity deberia ser mientras mas lejos menor valor, mas cerca mas valor (magnitud)
    pred_d1 = predicted_disparity[yp1, xp1]
    pred_d2 = predicted_disparity[yp2, xp2]
    print("pred_d1 ={} \n"
          "pred_d2 = {} ".format(pred_d1, pred_d2)
          )

    # # mean woman girth 80cm
    # smpl_z1 = z_anks[0]
    # smpl_z2 = z_anks[1]
    # pred_z1 = 1 / pred_d1
    # pred_z2 = 1 / pred_d2
    # scale = (smpl_z2 - smpl_z1) / (pred_z2 - pred_z1)
    # print('scale=%f'%scale)
    # t2 = smpl_z1 - scale * pred_z2
    # t1 = smpl_z2 - scale * pred_z1


def match_joints(image, folder, joint_names, all_gt_keypoints, persons_mask=None):
    img_shape = torch.Tensor([832, 512])[None]
    dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()
    h, w, _ = image.shape
    smpl_n = len(joint_names)
    detectron_n = len(all_gt_keypoints)
    if debug1:
        print('persons detected by SMPL={}, DEtectron={}'.format(smpl_n, detectron_n))
    # read all SMPL joints
    _, joints_trans_all, _ = read_all_joints(folder, joint_names)
    joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
    all_joints_projected = project_joints_to_img(joints_trans_all_t, img_shape, dummy_trans)
    all_joints_projected = all_joints_projected.cpu().numpy()
    # plot_joints_cv2(image, all_joints_projected)

    # plot_joints_cv2(persons_mask[..., None] * image, all_joints_projected)
    smpl_projected_int = all_joints_projected.astype(int)
    smpl_joints_as_idx = smpl_projected_int.reshape([-1, 2])

    xs, ys = zip(*smpl_joints_as_idx)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    smpl_joints_as_idx = np.array([ys, xs]).T

    # this is to filter initial smpl overestimated number of people in obviuos cases
    joint_in_mask = []
    for idx in smpl_joints_as_idx:
        mask_at_joint = persons_mask[tuple(idx)]
        joint_in_mask.append(mask_at_joint)
    joint_in_mask = np.array(joint_in_mask)
    joint_in_mask = joint_in_mask.reshape([-1, 24])
    pers_in_mask = joint_in_mask.sum(1) > 15
    overestimated_smpl = len(all_joints_projected) > pers_in_mask.sum()

    glb_vis = (all_gt_keypoints[..., -1] > 0.0).float()[0].cpu().numpy()
    dist = vectorize_distance(glb_vis[None, :, None] * all_joints_projected[..., :2],
                              all_gt_keypoints[..., :2].cpu().numpy()
                              )
    # first is for pred_kp_scl, second for glob_kpts
    idxs_pred, idxs_glob = linear_sum_assignment(dist)
    sorted_smpl_joints = all_joints_projected[idxs_pred]
    sorted_gt_kpts = all_gt_keypoints[idxs_glob].cpu().numpy()

    detectron_understimated = len(all_joints_projected) > len(all_gt_keypoints) and \
                              not overestimated_smpl
    if detectron_understimated:
        fixed_sorted_smpl_joints = sorted_smpl_joints.copy()
        fixed_detctions = []
        l_pred = len(idxs_pred)
        count = np.arange(0, l_pred + 1)
        which_missing = set(count) - set(idxs_pred)
        ones = np.ones_like(sorted_gt_kpts[0, :, :1])
        for missing in which_missing:
            print('missing:%d'%missing)
            fixed_sorted_smpl_joints = np.concatenate([sorted_smpl_joints,
                                                       all_joints_projected[missing][None]], 0)
            fixed_detctions.append(sorted_gt_kpts)
            to_add = np.concatenate([all_joints_projected[missing][None], ones[None]], 2)
            fixed_detctions.append(to_add)
            idxs_pred = np.concatenate([idxs_pred, [missing]])
        sorted_gt_kpts = np.concatenate(fixed_detctions)
        sorted_smpl_joints = fixed_sorted_smpl_joints

    if not persons_mask is None:
        return sorted_smpl_joints, sorted_gt_kpts, joints_trans_all_t, pers_in_mask, idxs_pred, idxs_glob
    else:
        return sorted_smpl_joints, sorted_gt_kpts, joints_trans_all_t, idxs_pred, idxs_glob

def remove_depth_outliers_from_plane(scaled_disparity_norm, road):
    outlier_idx = np.where(scaled_disparity_norm == scaled_disparity_norm.max())
    mean_val = scaled_disparity_norm.mean()
    scaled_disparity_norm[outlier_idx] = mean_val
    scaled_disparity_road = scaled_disparity_norm[road]
    return scaled_disparity_road


def get_absolute_nomalized_plane_hips_manual(image, folder, name, predicted_disparity,
                                        road, h, w, args, all_gt_keypoints, joint_names,
                                             hip_width=0.23):
    """
    desambiguando con la distancia de los hips.
    usando los joints de hips detectados con el detector de 2d kpts porque la proyeccion de SMPL
    puede tener pose con rotacion no tan buena.

    Para obtener el plano incicial y que he usado en midasv2 (anterior codigo) lo que hago es
    normalizar los pixeles de -1 a 1 invirtiendo el signo de disparity *-1 y luego haciendo shift
    hacia el ankle de referencia.

    Ahora, si quiero tener la escala absoluta debo hacer que se correspondan los pixeles de hip con
    las hips de la persona en coordenadas mundo? Creo que no, porque solo uso la relacion de
    differencia entre valores de disparity en cada punto y valores de depth reales en cada punto.
    Diria que no porque lo que cambia la escala es un cambio en z y no en xy.

    Aqui digo depth porque al inversit el signo del disparity (ojo no invertir el valor haciendo 1/x)
    quiero que luego el disparity haga match con el depth preservando la relacion lineal.
    """

    out_dir = args.output_path
    out_dir = out_dir.replace('output', 'results')
    iname = name.replace('_3djoints_0.json', '')

    ankles_, ankles_translated, translation_ = read_ankles(folder, name)

    img_shape = torch.Tensor([832, 512])[None]
    inspect_3d_joints(image, folder, name, img_shape)

    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)

    # todo determinar la escala con info del hip

    scale = -1
    t = 0
    plot(predicted_disparity)
    plot(road * predicted_disparity)
    predicted_disparity_norm = 2*(predicted_disparity / 65535) - 1
    # 16bit resolution
    scaled_disparity_norm = scale * predicted_disparity_norm + t

    sorted_smpl_joints, sorted_gt_kpts, joints_trans_all_t = match_joints(image,
                                                                          folder,
                                                                          joint_names,
                                                                          all_gt_keypoints)
    plot_joints_cv2(image, sorted_gt_kpts)
    # get hips
    all_hips = sorted_gt_kpts[:, [2, 3]]
    kpts_conf = sorted_gt_kpts[..., 2]
    plot_joints_cv2(image, all_hips)
    plot_joints_cv2(predicted_disparity, all_hips)
    xy_hips = all_hips[..., :2].astype(np.int)
    hips_conf = all_hips[..., 2]
    # perspective projected ankles, to get the img coords and take the disp values
    idx_pers = 1
    # these are openpose joints
    plot_joints_cv2(image, sorted_gt_kpts[idx_pers][None], with_text=True)
    these_hips_conf = hips_conf[idx_pers]
    these_pers_conf = kpts_conf[idx_pers]
    xp1, yp1 = xy_hips[idx_pers, 0]
    xp2, yp2 = xy_hips[idx_pers, 1]
    plot_joints_cv2(predicted_disparity, xy_hips[idx_pers, [0, 1]][None], with_text=True)

    # disparity deberia ser mientras mas lejos menor valor, mas cerca mas valor (magnitud)
    pred_d1 = predicted_disparity[yp1, xp1]
    pred_d2 = predicted_disparity[yp2, xp2]
    print("pred_d1 ={} \n"
          "pred_d2 = {} ".format(pred_d1, pred_d2)
          )

    scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_norm, road)

    w_range_norm = (2 * w_range / w) - 1
    h_range_norm = (2 * h_range / h) - 1

    x_world_norm =  w_range_norm[road]
    y_world_norm = h_range_norm[road]
    q_3d = np.stack([x_world_norm,  y_world_norm, scaled_disparity_road], 1)

    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                '%s_estimated_plane_normalized.ply'%iname ),
                                               return_normal=True
                                 )

    # todo usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]
    plane = eval_plane(a,b,c,-d, xrange, yrange)
    out_name = os.path.join(out_dir,'%s_shifted_estimated_plane_norm.ply' % (iname))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'%out_name)

    # todo scale with hips info
    # mean woman girth 80cm
    # delta_real_z = 0.80
    delta_real_z = hip_width
    delta_real_z = 0.18
    # take pred_dips from normalized disparity, not directly from model's output
    pred_d1_norm = predicted_disparity_norm[yp1, xp1]
    pred_d2_norm = predicted_disparity_norm[yp2, xp2]
    print("pred_norm_d1 ={} \n"
          "pred_norm_d2 = {} ".format(pred_d1_norm, pred_d2_norm)
          )

    abs_scale = delta_real_z / (pred_d2_norm - pred_d1_norm)
    print('scale=%f' % abs_scale)
    # scaled_disparity_abs = abs_scale * predicted_disparity_norm
    # scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_abs, road)

    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [lim/2, lim/1.2]
    q_3d = np.stack([x_world_norm,  y_world_norm, abs_scale * scaled_disparity_road], 1)
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                '%s_estimated_plane_absolute.ply'%iname ),
                                               return_normal=True
                                 )

    # todo usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    yrange = [lim/4, lim/1.5]
    plane = eval_plane(a, b, c, -d, xrange, yrange)
    out_name = os.path.join(out_dir, '%s_shifted_estimated_plane_abs.ply' % (iname))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'% out_name)

    return normal_vect, point0


def build_camera_instrinsic(camera_center, focal_length=1000):
    batch_size = 1
    K = np.zeros([batch_size, 3, 3], dtype=np.float32)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    k_inv = np.linalg.inv(K)
    return K, k_inv


def get_absolute_plane_hips_z(image, folder, name, predicted_disparity,
                                             road, h, w, args, all_gt_keypoints, joint_names,
                                             hip_width=0.23):
    """
    parecida a la anterior pero aqui escalo xy de acuerdo a la persona de referencia y luego
    trato de obtener delta z de los dos joints utilizados para obtener la longitud, ya sea
    hips u otros. En realidad lo que necesito es la magnitud de la proyeccion de la anchura del hip
    sobre el eje z. La anterior toma en cuenta solo la anchura del hip y no la proyeccion, lo que
    esta mal!
    """



    out_dir = args.output_path
    out_dir = out_dir.replace('output', 'results')
    iname = name.replace('_3djoints_0.json', '')

    ref_person = 1
    name = joint_names[ref_person]
    ankles_, ankles_translated, translation_ = read_ankles(folder, name)

    img_shape = torch.Tensor([832, 512])[None]
    inspect_3d_joints(image, folder, name, img_shape)

    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)

    # todo determinar la escala con info del hip
    scale = -1
    t = 0
    plot(predicted_disparity)
    plot(road * predicted_disparity)
    predicted_disparity_norm = 2*(predicted_disparity / 65535) - 1
    # 16bit resolution
    scaled_disparity_norm = scale * predicted_disparity_norm + t

    # joints_trans_all_t are SMPL in 3D
    sorted_smpl_joints, sorted_gt_kpts, joints_trans_all_t = match_joints(image,
                                                                          folder,
                                                                          joint_names,
                                                                          all_gt_keypoints)
    plot_joints_cv2(image, sorted_gt_kpts)
    # get hips
    all_hips = sorted_gt_kpts[:, [2, 3]]
    kpts_conf = sorted_gt_kpts[..., 2]
    plot_joints_cv2(image, all_hips)
    plot_joints_cv2(predicted_disparity, all_hips)
    xy_hips = all_hips[..., :2].astype(np.int)
    hips_conf = all_hips[..., 2]
    # perspective projected ankles, to get the img coords and take the disp values
    idx_pers = 1
    # these are openpose joints
    plot_joints_cv2(image, sorted_gt_kpts[idx_pers][None], with_text=True)
    these_hips_conf = hips_conf[idx_pers]
    these_pers_conf = kpts_conf[idx_pers]
    xp1, yp1 = xy_hips[idx_pers, 0]
    xp2, yp2 = xy_hips[idx_pers, 1]
    plot_joints_cv2(predicted_disparity, xy_hips[idx_pers, [0, 1]][None], with_text=True)

    # disparity deberia ser mientras mas lejos menor valor, mas cerca mas valor (magnitud)
    pred_d1 = predicted_disparity[yp1, xp1]
    pred_d2 = predicted_disparity[yp2, xp2]
    print("pred_d1 ={} \n"
          "pred_d2 = {} ".format(pred_d1, pred_d2)
          )

    scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_norm, road)

    # poner xy en el rango [-1, 1]
    w_range_norm = (2 * w_range / w) - 1
    h_range_norm = (2 * h_range / h) - 1
    x_world_norm = w_range_norm[road]
    y_world_norm = h_range_norm[road]
    q_3d = np.stack([x_world_norm, y_world_norm, scaled_disparity_road], 1)

    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                 '%s_estimated_plane_normalized.ply' % iname),
                                 return_normal=True )

    # usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]
    plane = eval_plane(a,b,c,-d, xrange, yrange)
    out_name = os.path.join(out_dir,'%s_shifted_estimated_plane_norm.ply' % (iname))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'%out_name)


    # get the z diff from body measurements
    plot_joints_cv2(image, sorted_smpl_joints)
    camera_center = img_shape.numpy() / 2
    K, k_inv = build_camera_instrinsic(camera_center)

    ones = np.ones_like(sorted_smpl_joints[..., 0])[..., None]
    zeros = np.zeros_like(sorted_smpl_joints[..., 0])[..., None]
    joints_homog = np.concatenate([sorted_smpl_joints, ones], 2)
    joints_homog = joints_homog.transpose(0, 2, 1)
    smpl_joints_imgSpace = k_inv @ joints_homog
    joints_world = smpl_joints_imgSpace.transpose(0, 2, 1)
    joints_world = joints_world[..., :2]
    joints_world = np.concatenate([joints_world, ones], 2)
    joints_world = joints_world[ref_person]
    # joints_world = joints_world.reshape([-1, 3])
    point0, point1 = ankles_translated
    x,y,z = point0
    joints_world = z * joints_world
    save_points(joints_world, './joints_world.ply')

    joints_smpl = joints_trans_all_t[ref_person].cpu().numpy()
    hips_smpl_est = joints_smpl[[2,3]]
    hips_w_smplEst = get_distance(hips_smpl_est)
    hips_smpl_est_2d = hips_smpl_est[..., :2]
    hips_d_smpl_2d = get_distance(hips_smpl_est_2d)
    save_points(joints_smpl, './joints_SMPL_world.ply')

    hipsXY_world_scale = joints_world[[2, 3], :2]

    hipsXY_world_scale = hips_smpl_est_2d
    hips_d_2d = get_distance(hipsXY_world_scale)
    # get depth between hips from mean measurements
    # hip_mean_dist = female_fields['hips_distances']
    hip_mean_dist = hips_d_2d
    dxy = abs(hipsXY_world_scale[0] - hipsXY_world_scale[1])
    dxy_sqr = (dxy*dxy).sum()
    hip_d_2 = hip_mean_dist*hip_mean_dist
    delta_z = np.sqrt(hip_d_2 - dxy_sqr)

    # todo scale with hips info
    # mean woman girth 80cm
    # delta_real_z = 0.80
    delta_real_z = hip_width
    delta_real_z = 0.18
    # take pred_dips from normalized disparity, not directly from model's output
    pred_d1_norm = predicted_disparity_norm[yp1, xp1]
    pred_d2_norm = predicted_disparity_norm[yp2, xp2]
    print("pred_norm_d1 ={} \n"
          "pred_norm_d2 = {} ".format(pred_d1_norm, pred_d2_norm)
          )

    abs_scale = delta_real_z / (pred_d2_norm - pred_d1_norm)
    print('scale=%f' % abs_scale)
    # scaled_disparity_abs = abs_scale * predicted_disparity_norm
    # scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_abs, road)


    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [lim/2, lim/1.2]
    q_3d = np.stack([x_world_norm,  y_world_norm, abs_scale * scaled_disparity_road], 1)
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                '%s_estimated_plane_absolute.ply'%iname ),
                                               return_normal=True
                                 )


    # todo usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    yrange = [lim/4, lim/1.5]
    plane = eval_plane(a, b, c, -d, xrange, yrange)
    out_name = os.path.join(out_dir, '%s_shifted_estimated_plane_abs.ply' % (iname))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'% out_name)

    return normal_vect, point0



def get_param_plane_xy_real_coords(image, folder, name, predicted_disparity,
                                             road, h, w, args, all_gt_keypoints, joint_names,
                                             h_scale=9):
    """
    escalar xy para que sea lo mas parecido a real coords
    parametrizar la escala de z del plano segun height de las personas
    luego probar la optmizacion para esto
    """
    focal_length = 1000
    out_dir = args.output_path
    out_dir = out_dir.replace('output', 'results')
    iname = name.replace('_3djoints_0.json', '')

    ref_person = 0
    name = joint_names[ref_person]
    ankles_, ankles_translated, translation_ = read_ankles(folder, name)

    img_shape = torch.Tensor([832, 512])[None]
    inspect_3d_joints(image, folder, name, img_shape)

    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)

    # todo determinar la escala con info del hip
    scale = -1
    t = 0
    plot(predicted_disparity)
    plot(road * predicted_disparity)
    predicted_disparity_norm = 2*(predicted_disparity / 65535) - 1
    # 16bit resolution
    scaled_disparity_norm = scale * predicted_disparity_norm + t

    # get hips
    scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_norm, road)
    # poner xy en el rango [-1, 1]
    w_range_norm = (2 * w_range / w) - 1
    h_range_norm = (2 * h_range / h) - 1
    x_world_norm = w_range_norm[road]
    y_world_norm = h_range_norm[road]


    # get the z diff from body measurements
    camera_center = img_shape.numpy() / 2
    K, k_inv = build_camera_instrinsic(camera_center)

    x_coords_f, y_coords_f = get_img_coords_flatten(w, h)
    ones = np.ones_like(x_coords_f)
    img_coords_hom = np.concatenate([x_coords_f, y_coords_f, ones], 1)
    world_coords = (k_inv[0] @ img_coords_hom.T).T
    z_ref = ankles_translated[0][2]
    world_coords_scaled = z_ref * world_coords
    save_points(world_coords_scaled, './inspect/img_world_coords_scaled.ply')

    # disp_shift = predicted_disparity - predicted_disparity.min()
    # disp_norm = disp_shift / disp_shift.max()
    # disp_norm = 2 * disp_norm - 1
    # s = z_ref / focal_length
    # disp_s = disp_shift * s / 10
    disp_norm_shifted = scaled_disparity_norm + z_ref
    disp_norm_shifted = disp_norm_shifted.reshape([-1, 1])
    q_3d = np.concatenate([world_coords_scaled[..., :2], disp_norm_shifted], 1)
    save_points(q_3d, './inspect/disp_world_coords.ply')

    w_range_norm_f = w_range_norm.reshape([-1, 1])
    h_range_norm_f = h_range_norm.reshape([-1, 1])
    cam_coords = np.concatenate([w_range_norm_f, h_range_norm_f], 1)
    q_3d = np.concatenate([cam_coords, disp_norm_shifted], 1)
    save_points(q_3d, './inspect/disp_norm_coords.ply')

    # todo scale with hips info
    # abs_scale controla la inclinacion en el eje z
    abs_scale = 1
    print('scale=%f' % abs_scale)
    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [lim/2, lim/1.2]
    q_3d = np.stack([x_world_norm,  y_world_norm, abs_scale * scaled_disparity_road], 1)
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                '%s_estimated_plane_absolute.ply'%iname ),
                                               return_normal=True
                                 )

    # todo usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    yrange = [lim/4, lim/1.5]
    plane = eval_plane(a, b, c, -d, xrange, yrange)
    out_name = os.path.join(out_dir, '%s_shifted_estimated_plane_abs.ply' % (iname))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'% out_name)

    return normal_vect, point0



def get_relative_nomalized_plane_v3(image, folder, name, predicted_disparity,
                                    road, h, w, args,  all_gt_keypoints,
                                    joint_names, ref_person=0, result_dir='', plane_scale=1.0,
                                    negative_plane=False,
                                    rotate_plane=False,
                                    w_sc=1.0,
                                    ):

    '''
    Use correction escale for all cases
    '''
    if debug1:
        print('ref person: %d' % ref_person)
    name = joint_names[ref_person]
    ankles_, ankles_translated, translation_ = read_ankles(folder, name)

    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)

    if negative_plane is False:
        scale = -1
    else:
        scale = 1
    t = 0
    # plot(predicted_disparity)
    if debug:
        plot(road * predicted_disparity)
    predicted_disparity_norm = 2*(predicted_disparity / 65535) - 1
    scaled_disparity_norm = scale * predicted_disparity_norm + t
    scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_norm, road)

    w_range_norm = (2 * w_range / w) - 1
    h_range_norm = (2 * h_range / h) - 1
    x_world_norm = w_range_norm[road]
    y_world_norm = h_range_norm[road]
    # w_range_norm_sc = 2 * w_range_norm

    # sc = 1.0
    # sc = 1.7
    # sc = 2.5
    # plane_scale = 8.0
    # w_sc = 4.0
    q_3d = np.stack([w_sc * x_world_norm, y_world_norm, plane_scale * scaled_disparity_road], 1)

    out_name = os.path.join(result_dir, 'disparity_points.ply')
    save_points(q_3d, out_name)
    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]

    out_plane_n = os.path.join(result_dir, 'estimated_plane_normalized.ply')
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                               name=out_plane_n,
                                               return_normal=True
                                               )

    # usa ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    # yrange = [lim/4, lim/1.5]
    yrange = [-lim, lim]
    plane, pend, angle = eval_plane(a, b, c, -d, xrange, yrange, return_pend=True)
    # print('pendiente: %f' % pend)
    out_name = os.path.join(result_dir, 'shifted_plane_norm.ply')
    save_points(plane, out_name)
    # print('plane pointcloud saved at: %s' % out_name)
    # print('normal vector = {} '.format(normal_vect))
    # print('normal vector = {} with scale {}'.format(normal_vect, h_scale))


    # rotate_plane = True
    if rotate_plane:
        # inspect normal vector
        x = np.linspace(0.93, 1., num=50)
        x0, y0, z0 = point0
        y = (b * (x - x0) / a) + y0
        z = (c * (x - x0) / a) + z0
        norm_vect_points = np.stack([x, y, z], 1)
        out_name = './norm_vect.ply'
        save_points(norm_vect_points, out_name)

        # rotate normal in z axis
        theta = -20
        theta = math.radians(theta)
        cos = math.cos(theta)
        sin = math.sin(theta)
        rotmat = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

        rot_normal = rotmat @ norm_vect_points.T
        rot_normal=rot_normal.T
        out_name = './norm_vect_rot.ply'
        save_points(rot_normal, out_name)

        rot_norm_vector = (rotmat @ normal_vect[:, None]).squeeze()
        a,b,c = rot_norm_vector
        plane, pend, angle = eval_plane(a, b, c, -d, xrange, yrange, return_pend=True)
        # print('pendiente: %f' % pend)
        out_name = os.path.join(result_dir, 'shifted_plane_norm_rotated.ply')
        save_points(plane, out_name)
        normal_vect = rot_norm_vector


    return normal_vect, point0, angle


def get_relative_nomalized_plane_param(image, folder, name, predicted_disparity,
                                    road, h, w, args,  all_gt_keypoints, joint_names,
                                    h_scale=9):

    '''
    Se podria seleccionar a la persona de referencia que tenga la altura mas realista,
    dado que no tiene una mala reproyeccion.
    Returns:

    '''
    out_dir = args.output_path
    out_dir = out_dir.replace('output', 'results')
    iname = name.replace('_3djoints_0.json', '')

    ref_person = 0
    name = joint_names[ref_person]
    ankles_, ankles_translated, translation_ = read_ankles(folder, name)

    img_shape = torch.Tensor([832, 512])[None]
    inspect_3d_joints(image, folder, name, img_shape)


    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)


    scale = -1
    t = 0
    plot(predicted_disparity)
    plot(road * predicted_disparity)
    predicted_disparity_norm = 2*(predicted_disparity / 65535) - 1
    scaled_disparity_norm = scale * predicted_disparity_norm + t
    scaled_disparity_road = remove_depth_outliers_from_plane(scaled_disparity_norm, road)

    w_range_norm = (2 * w_range / w) - 1
    h_range_norm = (2 * h_range / h) - 1
    x_world_norm = w_range_norm[road]
    y_world_norm = h_range_norm[road]
    q_3d = np.stack([x_world_norm,  y_world_norm, h_scale * scaled_disparity_road], 1)

    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                 name=os.path.join(out_dir,
                                '%s_estimated_plane_normalized_v2_s%d.ply'%(iname, h_scale)),
                                               return_normal=True
                                 )

    # todo usar ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    yrange = [lim/4, lim/1.5]
    plane = eval_plane(a, b, c, -d, xrange, yrange)
    out_name = os.path.join(out_dir, '%s_shifted_plane_param_s%f.ply' % (iname, h_scale))
    save_points(plane, out_name)
    print('plane pointcloud saved at: %s'% out_name)
    print('normal vector = {} with scale {}'.format(normal_vect, h_scale))



    return normal_vect, point0

def save_shifted_plane(normal_vect, point0, out_path):
    lim = 2
    a,b,c = normal_vect
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    # yrange = [lim/4, lim/1.5]
    yrange = [-lim, lim]
    plane = eval_plane(a, b, c, -d, xrange, yrange)
    save_points(plane, out_path)
    print('plane pointcloud saved at: %s'% out_path)



def main():
    args = get_args()
    default_models = {
        "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dataset = args.dataset
    # compute detections and depth maps
    if dataset=='paper':
        run_w_reproj_paper(args, args.input_path, args.output_path, args.model_weights,
                         args.model_type, args.optimize)
    elif dataset=='paper_w_det':
        run_w_det_paper(args, args.input_path, args.output_path, args.model_weights,
                         args.model_type, args.optimize)



if __name__ == "__main__":
    main()
