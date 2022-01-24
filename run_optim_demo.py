import sys
SEG_PATH = './external/panoptic_deeplab'
sys.path.insert(1, SEG_PATH)

import os
import glob
import torch
import numpy as np
import shutil
import tqdm
import pickle
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj

from util import plot_joints_cv2
from util.plane import  estimate_plane_xy_diff_range, eval_plane
from util.depth import read_ankles, read_all_joints, project_joints_to_img
from util.misc import save_points, draw_lsp_14kp__bone
from util.model import get_args
from util import vis_proj_joints_t, save_json

from losses import MSELoss, keypoint_loss, loss_ground_plane


debug = False
debug1 = False

def run(args, input_path):
    print("initialize")
    data_folder = ''
    img_dir_name = os.path.basename(input_path)
    dir_name = os.path.basename(os.path.dirname(data_folder))
    input_path = args.input_path
    img_names = glob.glob(os.path.join(input_path, "*"))
    img_names = [c for c in img_names if ('.jpg' in c or '.png' in c) and 'output.' not in c]
    img_names = [c for c in img_names if '_resized_resized' not in c]
    img_names = [c for c in img_names if 'final_reproj' not in c]
    subs_name = ''
    img_names.sort()

    num_images = len(img_names)

    plane_scale = args.plane_scale
    output_path = os.path.join(args.output_path, dir_name)
    negative_plane = args.negative_plane
    w_ordinal = args.w_ordinal_loss
    n_iters = args.n_iters
    rotate_plane = args.rotate_plane
    horizontal_scale = args.horizontal_scale
    w_reg_size = args.w_reg_size

    print('imgs todo: %d' % num_images)
    print('Doing folder: %s' % img_dir_name)
    print('mode: %s' % args.mode)
    print('')
    print('ordinal weight: %d' % args.w_ordinal_loss)
    print('number of iters: %d' % n_iters)


    out_dir = ''
    if args.mode == 'simple':
        out_dir = output_path.replace('output', '%sresults' % subs_name)
    elif args.mode == 'smpl_reproj':
        add_str = ''
        if rotate_plane:
            add_str = '_rot%d' % rotate_plane
        if horizontal_scale:
            add_str+='_wsc%.1f' % horizontal_scale
        if w_reg_size:
            add_str+='_regSC%.1f' % w_reg_size

        out_dir = output_path.replace('output', 'results')
        out_dir = os.path.join(out_dir,
                                      '%sresults_smplreproj_w%d_sc%.1f_its%d%s' % (subs_name,
                                                                                 w_ordinal,
                                                                                 plane_scale,
                                                                                 n_iters,
                                                                                 add_str))




    print('output folder: %s' % out_dir)
    for ind, img_name in tqdm.tqdm(enumerate(img_names), total=num_images):
        args.input_path = os.path.dirname(img_name)
        iname = os.path.basename(img_name).split('.')[0]
        name = '%s_3djoints_0.json' % iname
        image = Image.open(img_name)
        image = np.array(image)
        image = image[..., :3]

        # output
        folder = args.input_path
        joint_files = os.listdir(folder)
        joint_names = [c for c in joint_files if iname in c and '.json' in c]
        joint_names.sort()
        iname = name.replace('_3djoints_0.json', '')
        iname_strip = iname.split('_resized')[0]
        result_dir = os.path.join(out_dir, img_dir_name, iname)
        os.makedirs(result_dir, exist_ok=True)
        files = os.listdir(result_dir)
        for f in files:
            os.remove(os.path.join(result_dir, f))
        img_out = os.path.join(result_dir, 'image.jpg')
        plt.imsave(img_out, image)

        data_file = os.path.join('./precomputed_data', img_dir_name, iname, 'data_reproj_joints.pkl')
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        except:
            print('Data file not loaded: %s' % data_file)
            continue

        predicted_disparity = data['predicted_disparity']
        road = data['road']
        road_sum = road.sum()
        if road_sum==0:
            print(f'No road detected!! {img_dir_name}/{iname}')
            continue

        h = data['h']
        w = data['w']
        glob_kpts_all = data['glob_kpts_all']
        joint_names = data['joint_names']

        lsp_joints = glob_kpts_all[:, :14]
        img_w_j = draw_lsp_14kp__bone(image, lsp_joints[0])
        img_w_j = draw_lsp_14kp__bone(img_w_j, lsp_joints[1])
        plt.imsave('./test_w_skel.png', img_w_j)

        if args.mode == 'smpl_reproj':
            joint_opt_use_smpl_for_reproj(image, folder, name,
                                                iname_strip,
                                                 result_dir,
                                                 predicted_disparity, road, h, w, args,
                                                 glob_kpts_all, joint_names,
                                          plane_scale, negative_plane,
                                                 n_iters,
                                                rotate_plane,
                                                horizontal_scale
                                                 )

        else:
            print('mode not implemented!')
    print("finished")



def get_relative_nomalized_plane_v3(image, folder, name,
                                    predicted_disparity,
                                    road, h, w, args,
                                    all_gt_keypoints,
                                    joint_names,
                                    ref_person=0,
                                    result_dir='',
                                    plane_scale=1.0,
                                    negative_plane=False,
                                    rotate_plane=False,
                                    w_sc=1.0,
                                    ankles_translated=None,
                                    ):

    '''
    Use correction escale for all cases
    '''
    if debug1:
        print('ref person: %d' % ref_person)
    name = joint_names[ref_person]
    ankles_translated = ankles_translated[ref_person]
    if ankles_translated is None:
        _, ankles_translated, _ = read_ankles(folder, name)

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

    # un poco mas eficiente
    predicted_disparity_norm = 2*(predicted_disparity[road] / 65535) - 1
    scaled_disparity_road = scale * predicted_disparity_norm + t

    # mas eficiente
    x_world_norm = (2 * w_range[road] / w) - 1
    y_world_norm = (2 * h_range[road] / h) - 1

    # w_range_norm_sc = 2 * w_range_norm
    q_3d = np.stack([w_sc * x_world_norm, y_world_norm, plane_scale * scaled_disparity_road], 1)
    if debug1:
        out_name = os.path.join(result_dir, 'disparity_points.ply')
        save_points(q_3d, out_name)
    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]

    out_plane_n = os.path.join(result_dir, 'estimated_plane_normalized.ply')
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                               name=out_plane_n,
                                               return_normal=True,
                                               debug=debug
                                               )

    # usa ecuacion punto normal para trasladar el plano a los pie sde ref person
    a,b,c = normal_vect
    point0, point1 = ankles_translated
    d = np.dot(point0, normal_vect)
    xrange = [-2*lim, 2*lim]
    # yrange = [lim/4, lim/1.5]
    yrange = [-lim, lim]
    plane, pend, angle = eval_plane(a, b, c, -d, xrange, yrange, return_pend=True)

    return normal_vect, point0, angle


def joint_opt_use_smpl_for_reproj(image,
                                     folder,
                                     name,
                                     iname_strip,
                                     result_dir,
                                     predicted_disparity,
                                     road, h, w, args, all_gt_keypoints,
                                     joint_names,
                                     plane_scale, negative_plane, n_iters, rotate_plane,
                                     horizontal_scale
                                    ):
    """

    """

    obj_filename = '%s/%s_TRANS_person*.obj' % (folder, iname_strip)
    ori_meshes_n = glob.glob(obj_filename)
    ori_meshes_n.sort()
    for meshn in ori_meshes_n:
        shutil.copy(meshn, result_dir)

    criterion = MSELoss(reduction='none')

    img_shape = torch.Tensor([832, 512])[None]
    dummy_trans = torch.Tensor([[0., 0., 0.]]).cuda()
    # read all joints
    _, joints_trans_all, trans_all = read_all_joints(folder, joint_names)
    joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
    all_joints_projected = project_joints_to_img(joints_trans_all_t, img_shape, dummy_trans)
    ref_person = 1
    ankles_translated = joints_trans_all[:, [0, 5]]
    normal_vect, point0, pend = get_relative_nomalized_plane_v3(image, folder, name,
                                                                  predicted_disparity,
                                                                  road, h, w, args,
                                                                  all_gt_keypoints,
                                                                  joint_names,
                                                                  ref_person,
                                                                  result_dir,
                                                                  plane_scale,
                                                                  negative_plane,
                                                                  rotate_plane,
                                                                  horizontal_scale,
                                                                  ankles_translated
                                                                 )


    smpl_idxs_ = np.arange(0, len(all_joints_projected))
    idx_ref_person = np.where(smpl_idxs_ == ref_person)
    smpl_idxs = np.delete(smpl_idxs_, idx_ref_person)
    joints_smpl_3d_sorted = joints_trans_all_t[smpl_idxs]
    sorted_gt_kpts = all_joints_projected[smpl_idxs]
    ones_t = torch.ones_like(sorted_gt_kpts[:, :, 0])
    gt_keypoints = torch.cat([sorted_gt_kpts, ones_t[..., None]], 2)

    if debug:
        plot_joints_cv2(image, sorted_gt_kpts.cpu())
        plot_joints_cv2(image, sorted_gt_kpts[0, None].cpu())
        plot_joints_cv2(image, sorted_gt_kpts[1, None].cpu())
        plot_joints_cv2(image, sorted_gt_kpts[2, None].cpu())


    ankles = joints_smpl_3d_sorted[:, 0].cpu().numpy()
    # intersection btw trans line and plane, shift to plane
    trans_dir = ankles
    mod = np.linalg.norm(trans_dir, axis=1)
    trans_dir_ = trans_dir / mod[:, None]

    inter_d = np.dot(point0, normal_vect) / np.dot(trans_dir_, normal_vect)
    inter_pnt = inter_d[:, None] * trans_dir_
    trans_to_inter = inter_pnt - ankles
    trans_iter_t = torch.tensor(trans_to_inter, device='cuda')

    root_joint_t = joints_smpl_3d_sorted[:, 14, None, :]
    joints_p1_t = joints_smpl_3d_sorted

    normal_vect_t = torch.Tensor(normal_vect).cuda()
    point0_t = torch.Tensor(point0).cuda()

    # this are the ones to opt
    opt_trans = torch.tensor(trans_to_inter[:, None, :], requires_grad=True, device='cuda', dtype=torch.float32)
    opt_scale = torch.ones_like(trans_iter_t[:, 0, None, None], requires_grad=True, device='cuda', dtype=torch.float32)

    # create init optimizer
    lr = 0.01
    optimizer = torch.optim.Adam([opt_scale, opt_trans], lr=lr)
    print_every = 50000
    weight_plane = 100
    loss_reprojection = 10000
    loss = 10000
    # max_iters = 600 # for mupots TS3
    max_iters = n_iters
    w_loss_plane = 0

    crit = 1.0
    n_people = len(opt_scale)
    i = 0

    force = True
    while loss > crit or force:
        i += 1
        optimizer.zero_grad()
        # scale and trans
        trans_joints_p1_t = joints_p1_t + opt_trans
        scaled_joints_p1_t = opt_scale * (trans_joints_p1_t - root_joint_t) + root_joint_t
        joints_projected = project_joints_to_img(scaled_joints_p1_t, img_shape, dummy_trans)
        loss_reprojection, loss_raw = keypoint_loss(joints_projected, gt_keypoints, criterion)
        # get ankles
        ankles_p1_final_t = scaled_joints_p1_t[:, [0, 5]]
        loss_plane = loss_ground_plane(ankles_p1_final_t, normal=normal_vect_t, point=point0_t)
        w_loss_plane = weight_plane * loss_plane / n_people

        loss = w_loss_plane + loss_reprojection
        # backprop
        loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print('loss={}, loss_plane={}, loss_reproj={}'.format(loss,
                                         w_loss_plane,
                                         loss_reprojection,
                                         ))

        if i > max_iters:
            break

    out_losses = os.path.join(result_dir, '../losses_log.txt')
    print('loss={}, loss_plane={}, loss_reproj={}'.format(loss,
                                                             w_loss_plane,
                                                             loss_reprojection,
                                                             ), file=open(out_losses, "a"))

    new_scale = opt_scale.clone().detach()
    new_trans = opt_trans.clone().detach()
    final_joints_p1_t = joints_p1_t + new_trans
    final_joints_p1_t = new_scale * (final_joints_p1_t - root_joint_t) + root_joint_t
    final_joints_projected = project_joints_to_img(final_joints_p1_t, img_shape, dummy_trans)

    final_reproj_img = vis_proj_joints_t(image, final_joints_projected, gt_keypoints, do_plot=False)
    plt.imsave(result_dir + '/final_reproj.png', final_reproj_img)

    for ind, person_num in enumerate(smpl_idxs):
        obj_filename = '%s/%s_TRANS_person%d.obj' % (folder, iname_strip, person_num)
        verts, faces, _ = load_obj(obj_filename)
        faces_ = faces.verts_idx.numpy()
        verts_trans = verts.cuda() + new_trans[ind]
        verts_scaled = new_scale[ind] * (verts_trans - root_joint_t[ind]) + root_joint_t[ind]
        file = '%s/person%d_optim_stage2_w%d.obj' % (result_dir, person_num, weight_plane)
        out_mesh = trimesh.Trimesh(verts_scaled.cpu(), faces_, process=False)
        out_mesh.export(file);
        if debug1:
            print('final mesh saved at: %s' % file)



    scales_one = np.ones([n_people + 1], dtype=np.float32)
    scales = np.ones([n_people + 1], dtype=np.float32)
    scales[smpl_idxs] = new_scale[:, 0, 0].cpu().numpy()

    translations = np.zeros([n_people + 1, 1, 3], dtype=np.float32)
    translations[smpl_idxs] = new_trans.cpu().numpy()

    optim_joints_proj = np.zeros([n_people + 1, 24, 2], dtype=np.float32)
    optim_joints_proj[smpl_idxs] = final_joints_projected.cpu().numpy()
    all_joints_projected = all_joints_projected.cpu().numpy()
    optim_joints_proj[ref_person] = all_joints_projected[ref_person]
    optim_joints_3d = np.zeros([n_people + 1, 24, 3], dtype=np.float32)
    optim_joints_3d[smpl_idxs] = final_joints_p1_t.cpu().numpy()
    optim_joints_3d[ref_person] = joints_trans_all[ref_person]

    if debug:
        j3d_to_save = optim_joints_3d.reshape([-1, 3])
        save_points(j3d_to_save, './j3d_inspect.ply')
        save_points(optim_joints_3d[0], './j3d_inspect_0.ply')
        save_points(optim_joints_3d[1], './j3d_inspect_1.ply')
        save_points(optim_joints_3d[2], './j3d_inspect_2.ply')

        save_points(joints_trans_all[0], './j3d_inspect_crmp_0.ply')
        save_points(joints_trans_all[1], './j3d_inspect_crmp_1.ply')
        save_points(joints_trans_all[2], './j3d_inspect_crmp_2.ply')

    translations_final = trans_all + scales[:, None] * translations.squeeze()

    opt_info = {
        'scale': [scales.tolist()],
        'slope': [pend],
        'translations': translations_final.tolist(),
        'joints2d': optim_joints_proj.tolist(),
        'joints3d': optim_joints_3d.tolist(),
    }
    opt_info_file = '%s/optim_result.json' % (result_dir)
    save_json(opt_info_file, opt_info)

    opt_info = {
        'scale': [scales_one.tolist()],
        'translations': trans_all.tolist(),
        'joints2d': all_joints_projected.tolist(),
        'joints3d': joints_trans_all.tolist(),
    }
    opt_info_file = '%s/init_smpl_estim.json' % (result_dir)
    save_json(opt_info_file, opt_info)






def main(args):
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
    # compute depth maps
    run(args, args.input_path)

if __name__ == "__main__":
    args = get_args()
    main(args)
