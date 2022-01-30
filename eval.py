from util.misc import read_json
from util.eval_util import *
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.misc import mask_joints_w_vis
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
import pandas as pd

joints17_idx = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 0, 14, 1, 15, 16]
joints_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18]

ANNOTS = {
    #p_img, p_annots
'TS1':[3,2],
'TS2':[3,2],
'TS3':[3,2],
'TS4':[3,2],
'TS5':[3,2],
'TS6':[2,2],
'TS7':[3,3],
'TS8':[3,2],
'TS9':[3,2],
'TS10':[3,2],
'TS11':[2,2],
'TS12':[2,2],
'TS13':[3,3],
'TS14':[3,3],
'TS15':[3,3],
'TS16':[3,3],
'TS17':[3,3],
'TS18':[3,3],
'TS19':[3,3],
'TS20':[3,3],
}

path_names = {

    'all_ours_reproj': './results_smplreproj_w0_sc6.0_its600_wsc1.0/mupots-3d/',
    'frank_raw': './results_frankmocap/subsample_results_FRANKMOCAP_w0_sc6.0_its600_wsc1.0/mupots-3d/',

    'all_w_gt_rep': './results/mupots/results_GTreproj_w0_sc6.0_its600_wsc1.0/mupots-3d/',
    'all_only_rep_forceit': './results/mupots/results_ABLATION_w0_sc6.0_its600_reprjLoss_force_iters/mupots-3d/',
    'all_only_plane_forceit': './results/mupots/results_ABLATION_w0_sc6.0_its600_planeLoss_force_iters/mupots-3d/',

    'frank_all_ours_reproj' : './results_frankmocap_kada/better_reproj/results_FRANKMOCAP_w0_sc6.0_its600_wsc1.0_force_iters/mupots-3d',

}

GT_HEIGHT_SIGNS = {
    # median heigths for Mupots dataset, as there is no consistency frame by frame when
    # measured with limbs in a piece-wise manner
    'TS1': [1.82, 1.8],
    'TS2': [1.82, 1.8],
    'TS3': [1.82, 1.8],
    'TS4': [1.82, 1.8],
    'TS5': [1.82, 1.8],

    'TS6': [1.64, 1.64],
    'TS7': [1.59, 1.72, 1.43],

    'TS8': [1.64, 1.70],
    'TS9': [1.64, 1.70],
    'TS10': [1.64, 1.70],
    'TS11': [1.73, 1.71],
    'TS12': [1.75, 1.71],

    'TS13': [1.70, 1.68, 1.63],
    'TS14': [1.77, 1.71, 1.70],
    'TS15': [1.73, 1.71, 1.65],

    'TS16': [1.85, 1.69, 1.48],
    'TS17': [1.85, 1.69, 1.48],
    'TS18': [1.85, 1.69, 1.48],
    'TS19': [1.85, 1.69, 1.48],
    'TS20': [1.85, 1.69, 1.48],
}


class Mupots3D_v2(Dataset):
    def __init__(self, with3d=False, folder=None, evaluators=None, sanity=False,
                 match_lower=False, fmocap=False):
        root_d = '/home/nugrinovic/datasets/SMPL_MultiP_datasets/mupots-3d'
        annot = '/home/nugrinovic/datasets/SMPL_MultiP_datasets/mupots-3d/files/eval.pkl'
        self.root_d = root_d
        self.img_dir = root_d
        # load annot file
        f = annot
        with open(f, 'rb') as data:
            annots = pickle.load(data)

        img_paths = []
        joints2d = []
        joints3d = []
        joints3d_trans = []
        # len of annots is 8370
        if folder is None:
            d_folders = os.listdir(os.path.join('./input/mupots-3d/'))
            d_folders.sort()
            in_subjects = d_folders
        else:
            in_subjects = [folder]
        for ann in annots:
            filename = ann['filename']
            dir_name = os.path.basename(os.path.dirname(filename))
            if dir_name in in_subjects:
                img_paths.append(ann['filename'])
                joints2d.append(ann['kpts2d'].astype(np.float32))
                joints3d.append(ann['kpts3d'].astype(np.float32))
                joints3d_trans.append(ann['kpts3d_trans'].astype(np.float32))

        self.img_paths = np.array(img_paths)
        self.joints2d = np.array(joints2d)
        self.joints3d = np.array(joints3d)
        self.joints3d_trans = np.array(joints3d_trans)

        if sanity:
            length = 100
            self.img_paths = self.img_paths[:length]
            self.joints2d = self.joints2d[:length]
            self.joints3d = self.joints3d[:length]
            self.joints3d_trans = self.joints3d_trans[:length]

        self.data = annot
        self.no_json = 0
        self.no_slope = 0
        self.with3d = with3d
        self.evaluators = evaluators
        self.missing_files = []
        self.missing_baselines_files = []
        self.error_msg = []
        self.n_missing = 0
        self.match_lower = match_lower
        self.fmocap = fmocap
        self.folder = folder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        evaluators = self.evaluators
        match_lower = self.match_lower
        fmocap = self.fmocap
        folder = self.folder
        with3d = self.with3d
        f = self.img_paths[idx]
        dir = os.path.basename(os.path.dirname(f))
        iname = os.path.basename(f).split('.')[0]
        input_path = os.path.join('./input/mupots-3d', dir)
        orig_img = plt.imread(f)
        img_path = os.path.join(input_path, iname + '_resized.jpg')
        img = plt.imread(img_path)

        kpts2d = self.joints2d[idx]
        kpts3d = self.joints3d[idx]
        kpts3d_trans = self.joints3d_trans[idx]
        trans_gt = kpts3d_trans[:, 14, :3]


        baseDir = './results_baseline/results_smplreproj/mupots-3d/'
        resDir = './results_smplreproj_w0_sc6.0_its600_wsc1.0/mupots-3d/'

        if not fmocap:
            estimDir = './results_smplreproj_w0_sc6.0_its600_wsc1.0/mupots-3d/'
            estimPath = os.path.join(estimDir, dir, iname + '_resized', 'init_smpl_estim.json')
        else:
            estimDir = './results_frankmocap/subsample_results_FRANKMOCAP_w0_sc6.0_its600_wsc1.0/mupots-3d/'
            estimPath = os.path.join(estimDir, dir, iname + '_resized', 'init_smpl_estim.json')

        baselinePath = os.path.join(baseDir, dir, iname, 'optim_result.json')
        resPath = os.path.join(resDir, dir, iname + '_resized', 'optim_result.json')

        paths = []
        for ev in evaluators:
            if ev.name == 'frank_raw':
                this_path = os.path.join(path_names[ev.name], dir, iname + '_resized', 'init_smpl_estim.json')
            else:
                this_path = os.path.join(path_names[ev.name], dir, iname + '_resized', 'optim_result.json')
            paths.append(this_path)
        data = []
        try:
            initial_smpl = read_json(estimPath)
            baseline_res = read_json(baselinePath)
            results = read_json(resPath)

            for i, ev in enumerate(evaluators):
                this_data = read_json(paths[i])
                data.append(this_data)

        except Exception as e:
            self.no_json += 1
            self.n_missing += 1
            self.missing_files.append(f'{dir}/{iname}')
            return None

        try:
            slope_ours = data[0]['slope'][0]
        except Exception as e:
            slope_ours = 0
            self.no_slope += 1
            pass

        try:
            smpl_trans, j2d_initial, scale_initial, dict_initial = parse_data(initial_smpl, with3d)
            baseline_trans, j2d_baseline, scale_base, dict_base = parse_data(baseline_res, with3d)
            if np.isnan(j2d_baseline).any():
                self.n_missing += 1
                self.missing_files.append(f'{dir}/{iname}')
                return None
            out_trans, j2d_output, scale_ours, dict_ours = parse_data(results, with3d)
        except Exception as e:
            print(e)
            self.n_missing += 1
            self.missing_files.append(f'{dir}/{iname}')
            return None


        data_dicts = []
        for i, ev in enumerate(evaluators):
            try:
                _, _, _, this_data_dict = parse_data(data[i], with3d)
                data_dicts.append(
                    {ev.name:this_data_dict}
                )
            except Exception as e:
                print(e)
                self.n_missing += 1
                self.missing_files.append(f'{dir}/{iname}')
                return None

        dict_gt = {
            'scale': None,
            'translations': trans_gt,
            'joints2d': kpts2d,
            'joints3d': kpts3d,
            'heights': np.array(GT_HEIGHT_SIGNS[folder]),
        }

        optim_data = {
            'filename': f,

            'out_trans': out_trans,
            'smpl_trans': smpl_trans,
            'baseline_trans': baseline_trans,

            'j2d_output': j2d_output,
            'j2d_initial': j2d_initial,
            'j2d_baseline': j2d_baseline,

            'orig_img': orig_img,
            'slope_ours': slope_ours,

            'dict_gt': dict_gt,
            'dict_initial': dict_initial,
            'dict_base': dict_base,
            'dict_ours': dict_ours,
        }

        for d in data_dicts:
            optim_data.update(d)


        if match_lower:
            # pick the lowest n_people
            init_names = ['dict_gt', 'dict_initial', 'dict_base', 'dict_ours']
            ev_names = [c.name for c in evaluators]
            all_names = init_names + ev_names

            dicts_to_handle = [optim_data[c] for c in all_names]

            j2d_gt = mask_joints_w_vis(kpts2d)

            all_j2d = [c['joints2d'] for c in dicts_to_handle]

            npeopls = [c.shape[0] for c in all_j2d]
            npeopls = np.array(npeopls)
            minarg = np.argmin(npeopls)
            minp = np.min(npeopls)
            maxp = np.max(npeopls)

            h, w, _ = img.shape
            ho, wo, _ = orig_img.shape
            scalew = w / wo
            j2d_gt_sc = j2d_gt.copy()
            j2d_gt_sc[..., :2] = scalew * j2d_gt[..., :2]

            # do matiching with this as ref
            if maxp > minp:
                ref_j2d = all_j2d[minarg]
                bboxes_ref = bbox_from_joints_several(ref_j2d[..., :2])
                bboxes_gt = bbox_from_joints_several(j2d_gt_sc[..., :2])
                ids = np.arange(0, minp)
                iou_boxes, _ = box_iou_np(bboxes_ref, bboxes_gt)
                id_ref, id_gt = linear_sum_assignment(iou_boxes, maximize=True)
                id_gt.sort()
                ref_gt_j2d = j2d_gt_sc[id_gt]
                kpts2d = kpts2d[id_gt]
                bboxes_gt = bbox_from_joints_several(ref_gt_j2d[..., :2])
                #match the rest with this gt ref
                for i, j2d_this in enumerate(all_j2d):
                    bboxes_this = bbox_from_joints_several(j2d_this[..., :2])
                    iou_boxes, _ = box_iou_np(bboxes_this, bboxes_gt)
                    id_this, id_gt = linear_sum_assignment(iou_boxes, maximize=True)
                    id_this.sort()
                    # take out exceeding ids here
                    this_dict = dicts_to_handle[i]
                    this_dict['translations'] = this_dict['translations'][id_this]
                    this_dict['joints2d'] = this_dict['joints2d'][id_this]
                    this_dict['joints3d'] = this_dict['joints3d'][id_this]
                    if this_dict['scale'] is not None:
                        this_dict['scale'] = this_dict['scale'].squeeze()[id_this]

        return img, trans_gt, kpts2d, optim_data


class PairwiseOrderEvalv2():
    def __init__(self, name, is_gt=False):
        self.depths_lst = []
        self.gt_signs_l = []
        self.n_score = []
        self.lst_scores = []
        self.acc_score_gt = 0
        self.name = name

        self.is_gt = is_gt

    def run(self, this_trans, idxs_gt, gt_signs=None):
        # this does the depth order eval, all matrix operations
        name = self.name
        self.this_trans_filt = this_trans[idxs_gt]
        this_signs = get_sign_matix(self.this_trans_filt)
        if self.is_gt:
            gt_signs = this_signs
        pairwise_gt = np.equal(gt_signs, this_signs)
        scores_gt = upper_tri_masking(pairwise_gt)
        scores_gt_ = scores_gt.sum()
        self.acc_score_gt += scores_gt_
        self.depths_lst.append(self.this_trans_filt[:, 2])
        self.gt_signs_l.append(this_signs)
        self.lst_scores.append(scores_gt)
        self.n_score.append(scores_gt_)
        return this_signs

    def get_eval(self, acc_score_gt=None):
        res_dict = {
            'depths_%s' % self.name: self.depths_lst,
            'scores_raw_%s' % self.name: self.lst_scores,
        }

        if acc_score_gt is not None:
            percent_score = 100 * self.acc_score_gt / acc_score_gt
            return self.acc_score_gt, res_dict, percent_score
        else:
            return self.acc_score_gt, res_dict, None

    def reset(self):
        self.depths_lst = []
        self.gt_signs_l = []
        self.n_score = []
        self.lst_scores = []
        self.acc_score_gt = 0




def eval_depth_order_pairwise_v6(data_loader, evaluators, folder=None):

    print('-----eval_depth_order_pairwise-------')
    print('Folder: %s' % folder)
    folder_lst = []
    imgname_lst = []
    slope_ours_l = []

    p_img, p_annots = ANNOTS[folder]
    equal = p_img == p_annots

    gtEvaluator = PairwiseOrderEvalv2(name='gt', is_gt=True)
    EvaluatorInitial = PairwiseOrderEvalv2(name='crmp')
    EvaluatorBase = PairwiseOrderEvalv2(name='base')
    EvaluatorOurs = PairwiseOrderEvalv2(name='ours')

    for data in tqdm(data_loader):
        if data is None:
            continue
        img, gt_trans, j2d_gt, optim_data = data

        orig_img = optim_data['orig_img']
        filename = optim_data['filename']
        img_name = os.path.basename(filename)
        folder_name = os.path.basename(os.path.dirname(filename))

        # resize joints to CRMP img size
        h, w, _ = img.shape
        ho, wo, _ = orig_img.shape
        scalew = w / wo
        j2d_gt_sc = j2d_gt.copy()
        j2d_gt_sc[..., :2] = scalew * j2d_gt[..., :2]
        vis = j2d_gt_sc[0, :, 2][None, :, None]

        dict_gt = optim_data['dict_gt']
        dict_initial = optim_data['dict_initial']
        j2d_gt, gt_trans, sc_gt, j3d_gt = get_data_from_dict(dict_gt, with3D=True)
        j2d_initial, smpl_trans, sc_init, j3d_tgt = get_data_from_dict(dict_initial, with3D=True)
        idxs_initial, idxs_gt = order_idx_by_gt_j2d(vis, j2d_gt_sc, j2d_initial)


        def run_eval_data(img, data_dict, vis, j2d_gt_sc, gt_signs, evaluator, idxs_gt_ref):
            dict_initial = data_dict
            j2d_initial, trans, _ = get_data_from_dict(dict_initial)

            # first throw out bad assignations between detected 2d jts, based on iou
            idxs_initial, idxs_gt = order_idx_by_gt_j2d(vis, j2d_gt_sc, j2d_initial)
            trans = trans.squeeze()
            gt_sort = np.argsort(idxs_gt)
            idxs_gt = idxs_gt[gt_sort]
            idxs_initial = idxs_initial[gt_sort]

            # helps as check for bad person assignments (estimated vs. GT)
            if len(idxs_initial) != len(gt_signs):
                gt_signs = gt_signs[idxs_gt]
                gt_signs = gt_signs[:, idxs_gt]

            evaluator.run(trans, idxs_initial, gt_signs)

        dict_base = optim_data['dict_base']
        dict_ours = optim_data['dict_ours']

        # more assignments checks to be sure each person is compared with its corresponding GT,
        # if not better discard it
        rank, max_r, n_j2d_gt, n_j3d = get_ranks(j2d_gt_sc, j2d_initial, j3d_gt, j3d_tgt, vis)
        if rank==max_r and not equal:
            continue
        elif rank<p_img and equal:
            continue
        if n_j2d_gt>n_j3d:
            continue

        gt_sort = np.argsort(idxs_gt)
        idxs_gt = idxs_gt[gt_sort]

        gt_signs = gtEvaluator.run(gt_trans, idxs_gt)

        run_eval_data(img, dict_initial, vis, j2d_gt_sc, gt_signs, EvaluatorInitial, idxs_gt)
        run_eval_data(img, dict_base, vis, j2d_gt_sc, gt_signs, EvaluatorBase, idxs_gt)
        run_eval_data(img, dict_ours, vis, j2d_gt_sc, gt_signs, EvaluatorOurs, idxs_gt)

        for ev in evaluators:
            run_eval_data(img, optim_data[ev.name], vis, j2d_gt_sc, gt_signs, ev, idxs_gt)


        slope_ours = optim_data['slope_ours']
        slope_ours_l.append(slope_ours)
        folder_lst.append(folder_name)
        imgname_lst.append(img_name)
        pass

    acc_score_gt, res_dict_gt, _ = gtEvaluator.get_eval()
    acc_score_crmp, res_dict_crmp, perc_sc_crmp = EvaluatorInitial.get_eval(acc_score_gt)
    acc_score_base, res_dict_base, perc_sc_base = EvaluatorBase.get_eval(acc_score_gt)
    acc_score_ours, res_dict_ours, perc_sc_ours = EvaluatorOurs.get_eval(acc_score_gt)

    fixed_names = ['CRMP', 'baseline', 'ours']
    fixed_scores = [acc_score_crmp, acc_score_base, acc_score_ours]
    fixed_percs = [perc_sc_crmp, perc_sc_base, perc_sc_ours]

    scores = []
    dicts = []
    percs = []
    names = []
    for ev in evaluators:
        this_score, this_dict, this_perc = ev.get_eval(acc_score_gt)
        scores.append(this_score)
        dicts.append(this_dict)
        percs.append(this_perc)
        names.append(ev.name)


    not_included =  data_loader.dataset.no_json
    print('Data (files) NOT included: %d' % not_included)
    print('GT total pairs: %d' % acc_score_gt)
    print('CRMP score: %d, perc: %.2f' % (acc_score_crmp, perc_sc_crmp))
    print('baseline score: %d, perc: %.2f' % (acc_score_base, perc_sc_base))
    print('ours score: %d, perc: %.2f' % (acc_score_ours, perc_sc_ours))

    for i, ev in enumerate(evaluators):
        print('%s score: %d, perc: %.2f' % (ev.name, scores[i], percs[i]))

    print(" ")

    files_info = {
        'folder': folder_lst,
        'files': imgname_lst,
    }
    return names, scores, percs, fixed_names, fixed_scores, fixed_percs, files_info





def pair_depth_order_generic(evals_list, eval_name='generic', match_lower=False,
                             frank=False):

    d_folders = os.listdir(os.path.join('./input/mupots-3d/'))
    all_percs = []
    all_folders = []
    results_dir = f'./eval/mupots/depth_order/{eval_name}'
    os.makedirs(results_dir, exist_ok=True)
    for folder in d_folders:
        evaluators = [PairwiseOrderEvalv2(name=c) for c in evals_list]
        # dataset for loading results
        dataset = Mupots3D_v2(folder=folder, evaluators=evaluators, with3d=True,
                              sanity=False, match_lower=match_lower,
                              fmocap=frank)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print('Datset length: %d' % len(dataset))

        # run eval here
        names, scores, percs, fixed_names, fixed_scores, fixed_percs,\
        files_info = eval_depth_order_pairwise_v6(data_loader, evaluators, folder)

        folder_lst = files_info['folder']
        all_folders += folder_lst
        all_percs.append(fixed_percs + percs)

    # generate csv file with the results
    all_names = fixed_names + names
    summ = [all_names] + all_percs
    summ_np = np.array(summ).T
    df_summ = pd.DataFrame(summ_np)
    df_summ.to_csv(f'{results_dir}/percentages_eval_{eval_name}.csv')






def mains():
    # list of different evaluation results folders, one for each experiment
    evals_list = [
        'all_ours_reproj',
    ]
    pair_depth_order_generic(evals_list=evals_list, eval_name='all_data_depth_order')


if __name__ == '__main__':
    mains()


