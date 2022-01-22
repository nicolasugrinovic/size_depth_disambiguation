import torch
import cv2
import argparse
import sys
SEG_PATH = './external/panoptic_deeplab'
sys.path.insert(1, SEG_PATH)
from torchvision.transforms import Compose
from external.midas.dpt_depth import DPTDepthModel
from external.midas.midas_net import MidasNet
from external.midas.midas_net_custom import MidasNet_small
from external.midas.transforms import Resize, NormalizeImage, PrepareForNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path',
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path',
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights',
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type',
        default='dpt_large',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )


    ####### args de panoptic !
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--dset',
        default='coco',
        help='options: coco, mupots'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')

    parser.add_argument('--mode', dest='mode', default='simple', help='modes: simple, depth_order, '
                                                                      'plane_scale, smpl_reproj, '
                                                                      'frankmocap, use_j2d_gt, ablation')
    parser.add_argument('--w_ordinal_loss', default=0,  type=int)
    parser.add_argument('--plane_scale', default=1.0,  type=float)
    parser.add_argument('--negative_plane', action='store_true')
    parser.add_argument('--n_iters', default=600,  type=int)
    parser.add_argument('--w_plane', default=100,  type=int)
    parser.add_argument('--rotate_plane', default=0,  type=int)
    parser.add_argument('--horizontal_scale', default=1.0,  type=float)
    parser.add_argument('--do_subsample', action='store_true')
    parser.add_argument('--w_reg_size', default=0.0,  type=float)
    parser.add_argument('--better_reproj_fmocap', action='store_true')
    parser.add_argument('--force_iters', action='store_true')
    parser.add_argument('--dataset', default='mupots')
    parser.add_argument('--use_reprojection', action='store_true')
    parser.add_argument('--only_angles', action='store_true')
    parser.add_argument('--use_plane', action='store_true')
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--offset', default=0, type=int)
    parser.add_argument('--noise', default=0.0, type=float)

    parser.set_defaults(optimize=True)

    args = parser.parse_args()
    return args

def init_network(model_type, model_path, device, optimize):
    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize==True:
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
    model.to(device)

    return model, net_w, net_h, resize_mode, normalization, transform


def get_prediction(img, img_input, model, optimize, device):
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )
    return prediction