import sys
SEG_PATH = './external/panoptic_deeplab'
sys.path.insert(1, SEG_PATH)

import os
import glob
import torch
import utils
import tqdm
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from util.depth import read_all_joints, project_joints_to_img
from util.detectron import setup_cfg
from util.model import get_args, init_network, get_prediction

from detectron2.data.detection_utils import read_image
from external.panoptic_deeplab.tools_d2.d2.predictor import VisualizationDemo



def run_w_reproj_demo(args, model_path, model_type="large", optimize=True):
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    model, net_w, net_h, resize_mode, normalization, transform = init_network(model_type, model_path, device, optimize)

    # initialize
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # get input
    input_path = args.input_path
    img_names = glob.glob(os.path.join(input_path, "*"))
    img_names = [c for c in img_names if ('.jpg' in c or '.png' in c) and 'output.' not in c]
    img_names.sort()
    num_images = len(img_names)
    print('Number of images to process: %d' % num_images)
    dir_name = ''
    num_wrong_onePerson = 0
    output_path = os.path.join(args.output_path, dir_name)
    os.makedirs(output_path, exist_ok=True)
    print('Doing folder: %s' % dir_name)
    print('output folder: %s' % output_path)

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

        # using this output, its normalized btw [0, 65k]
        out_depth = out_depth_16bits
        zero_idxs = np.where(out_depth == 0.0)
        mean_disp = out_depth.mean()
        out_depth[zero_idxs] = mean_disp
        predicted_disparity = out_depth

        # use PIL, to be consistent with evaluation
        img = read_image(img_name, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        os.makedirs('./results', exist_ok=True)
        out_filename = './results/panoptic_%d.png' % np.random.randint(0, 1000)
        visualized_output.save(out_filename)

        pan = predictions['panoptic_seg'][0].cpu().numpy()
        pan = (pan / 1000).astype(int)
        labels = np.unique(pan)
        classes = []

        for l in labels:
            classes.append(demo.metadata.stuff_classes[l])
        # possible classes used as plane
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

        result_dir = os.path.join(output_path, iname)
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
        # so with this data, reprojection will be obtained from initial est.
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
    if dataset=='demo':
        run_w_reproj_demo(args, args.model_weights, args.model_type, args.optimize)
    elif dataset=='mupots':
        pass

    print('Done precomputing data')

if __name__ == "__main__":
    main()
