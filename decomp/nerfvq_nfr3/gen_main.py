import os
from os.path import join
import json
import numpy as np
from nerfactor.util import io as ioutil, config as configutil
import matplotlib.pyplot as plt

del_main_flag = True
best_thres = 0.002
# best_thres = 0.001, non-individual
use_baseline_nvq = False
baseline_nvq = {'ficus_2188': 4, 'drums_3072': 5, 'materials_2163': 7, 'lego_3072': 8, 'hotdog_2163': 6,
                'chair0_3072': 3, 'machine1_3072': 2, 'kitchen6_7095': 6}


def glob_dir(root, start_with=None, end_with=None, key=None, ):
    globed = []
    raw_fs = os.listdir(root)
    for f in raw_fs:
        if os.path.isdir(join(root, f)):
            glob_flag = True
            if start_with is not None:
                if not f[:len(start_with)] == start_with: glob_flag = False
            if end_with is not None:
                if not f[-len(end_with):] == end_with: glob_flag = False
            if key is not None:
                if key in f: glob_flag = True
                else: glob_flag = False
            if glob_flag: globed.append(f)
    return globed

out_root = '/home/zhonghongliang/vqnfr_pro_release/decomp/nerfvq_nfr3/output/train'
scene_roots = glob_dir(out_root, key='vq_nfr')
for scene_root in scene_roots:

    print(scene_root)
    scene_comps = scene_root.split('_')[:2]
    scene_name = '_'.join(scene_comps)
    if use_baseline_nvq and (not scene_name in baseline_nvq.keys()):
        print('Skip.')
        continue

    lr_fs = os.listdir(join(out_root, scene_root))
    lr = lr_fs[0].split('.')[0]
    config_ini = join(out_root, scene_root, lr+'.ini')
    config = ioutil.read_config(config_ini)

    num_embed = config.getint('DEFAULT', 'num_embed')
    num_drop = config.getint('DEFAULT', 'num_drop')
    x_list = list(range(num_embed - num_drop, num_embed + 1))

    if best_thres is None:
        best_thres = config.getfloat('DEFAULT', 'best_thres')

    epoch_dirs = glob_dir(join(out_root, scene_root, lr, 'vis_vali'), start_with='epoch')
    for epoch_dir in epoch_dirs:
        if epoch_dir == 'epoch000000150':
            vis_flag = True
        else: vis_flag = False

        # print('# Processing: ', scene_root, epoch_dir)
        scenr_epoch_root = join(out_root, scene_root, lr, 'vis_vali', epoch_dir)

        raw_vq_i_dirs = glob_dir(scenr_epoch_root)
        if del_main_flag:
            for vq_i_dir in raw_vq_i_dirs:
                if vq_i_dir[:5] == 'main_':
                    os.rename(join(scenr_epoch_root, vq_i_dir), join(scenr_epoch_root, vq_i_dir[5:]))
                vq_i_dirs = glob_dir(scenr_epoch_root)
        else:
            vq_i_dirs = raw_vq_i_dirs

        vq_ids = [int(vq_i) for vq_i in vq_i_dirs]
        vq_ids.sort()
        n_vq = len(vq_ids)
        if vis_flag: print('# Globbed VQ IDs ', vq_ids)

        # This file can be extracted from the vq ckpts using the evaluation codes in train_nfr.py
        with open(join(scenr_epoch_root, 'vq_test_loss.json')) as f:
            vq_test_scores = json.load(f)

        # if vis_flag: print('# VQ Range Num; ', n_vq, ' - VQ Score Num: ', len(vq_test_scores['vqrgb']))
        # chromaticity will exist in whatever condition
        # if vis_flag: print('# VQ Range Num; ', n_vq, ' - VQ Score Num: ', len(vq_test_scores['chromaticity']))

        drop_losses = []
        for i in range(n_vq):
            # drop_losses.append(vq_test_scores['vqrgb'][i] + vq_test_scores['chromaticity'][i])
            drop_losses.append(vq_test_scores['chromaticity'][i])
        drop_losses = np.array(drop_losses)

        plt.clf()
        plt.plot(x_list, drop_losses)
        plt.savefig(join(scenr_epoch_root, 'vq_num.png'))

        main_thres = best_thres

        for i in range(1, n_vq - 1):
            if drop_losses[i - 1] > drop_losses[i]:
                best_flag = True
                for j in range(i + 1, n_vq):
                    if drop_losses[i] - drop_losses[j] > main_thres:
                        best_flag = False
                        break
            else:
                best_flag = False
            if best_flag:
                main_vq = i
                break
        if not best_flag: main_vq = n_vq - 1

        if use_baseline_nvq: main_id = baseline_nvq[scene_name]
        else: main_id = vq_ids[main_vq]
        if vis_flag: print('# New Main VQ ID: ', main_id)
        os.rename(join(scenr_epoch_root, str(main_id)), join(scenr_epoch_root, 'main_' + str(main_id)))
