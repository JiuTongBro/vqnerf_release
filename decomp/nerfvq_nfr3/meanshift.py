import cv2
import numpy as np
from os.path import join
import torch
from sklearn.cluster import MeanShift
import os

'''
To run the Meanshift ablation, you can first train the continuous model (nfr_unit) to 300 epochs 
(keep the total training epochs the same as our decomposition model), then use the 'nerfactor/genz.sh' and this code.
'''

'''
data_roots = ['/home/zhonghongliang/vqnfr_pro/data/nfr_blender'] * 5 + \
            ['/home/zhonghongliang/vqnfr_pro/data/mat_blender'] * 3 + \
            ['/home/zhonghongliang/vqnfr_pro/data/dtu_split2'] * 3 + \
            ['/home/zhonghongliang/vqnfr_pro/data/1115_hw_data/1115data_1'] * 3 + \
            ['/home/zhonghongliang/vqnfr_pro/data/colmap_split'] * 3
scenes = ['drums_3072', 'lego_3072', 'hotdog_2163', 'ficus_2188', 'materials_2163',
          'chair0_3072', 'kitchen6_7095', 'machine1_3072',
          'dtu_scan24', 'dtu_scan69', 'dtu_scan110',
          'rabbit_-1', 'hwchair_-1', 'redcar_-1',
          'colmap_wshoes', 'colmap_bottle', 'colmap_tools2']
'''

data_roots = ['/home/zhonghongliang/vqnfr_pro/data/nfr_blender']
scenes = ['ficus_2188',]

pred_root = '/home/zhonghongliang/vqnfr_pro/nerfvq_nfr3/output/train'
dst_root = '/home/zhonghongliang/vqnfr_pro/tvcg_results/cluster_0428'

bandwidths = [0.5, 0.3, 0.2]
device_name = 'cpu'
n_samples = 10000

embed_c = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
           np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]),
           np.array([128, 0, 0]), np.array([0, 128, 0]), np.array([0, 0, 128]),
           np.array([128, 128, 0]), np.array([128, 0, 128]), np.array([0, 128, 128]),
           np.array([255, 128, 128]), np.array([128, 255, 128]), np.array([128, 128, 255]),
           np.array([255, 255, 128]), np.array([255, 128, 255]), np.array([128, 255, 255]),
           np.array([255, 128, 0]), np.array([255, 0, 128]), np.array([0, 255, 128]),]

color_map = {i+1: embed_c[i] for i in range(len(embed_c))}

for bandwidth in bandwidths:
    print(bandwidth)
    dst = join(dst_root, str(bandwidth))
    if not os.path.exists(dst): os.makedirs(dst)

    for data_root, scene in zip(data_roots, scenes):
        h, w = None, None
        dst_scene = join(dst, scene)
        if not os.path.exists(dst_scene): os.makedirs(dst_scene)

        print(scene)
        pred_scene = join(pred_root, scene + '_nfr_unit', 'lr5e-4', 'vis_z', 'ckpt-10')
        views = os.listdir(pred_scene)
        zs, val_zs, val_masks = [], [], []

        device = torch.device(device_name)
        for view in views:
            if view[:6] == 'train_':
                view_data = join(data_root, scene, view)
                view_pred = join(pred_scene, view)
                # print(view)
                gt_img = cv2.imread(join(view_data, 'rgba.png'), -1)/65535.

                z_d = cv2.imread(join(view_pred, 'albedo.png'), -1)[..., [2, 1, 0]] / 255.
                z_r = cv2.imread(join(view_pred, 'rough.png'), -1)[..., None] / 255.
                z_s = cv2.imread(join(view_pred, 'spec.png'), -1)[..., [2, 1, 0]] / 255.
                z = np.concatenate([z_d, z_s, z_r], axis=-1)

                if h is None: h, w = z_d.shape[:2]
                if gt_img.shape[0] != h:
                    gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)

                mask = gt_img[..., 3] > 0
                z_mask = z[mask]

                zs.append(torch.from_numpy(z_mask).to(device))



            elif view[:4] == 'val_':
                view_data = join(data_root, scene, view)
                view_pred = join(pred_scene, view)

                gt_img = cv2.imread(join(view_data, 'rgba.png'), -1)/65535.

                z_d = cv2.imread(join(view_pred, 'albedo.png'), -1)[..., [2, 1, 0]] / 255.
                z_r = cv2.imread(join(view_pred, 'rough.png'), -1)[..., None] / 255.
                z_s = cv2.imread(join(view_pred, 'spec.png'), -1)[..., [2, 1, 0]] / 255.
                z = np.concatenate([z_d, z_s, z_r], axis=-1)

                if h is None: h, w = z_d.shape[:2]
                if gt_img.shape[0] != h:
                    gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_AREA)

                mask = gt_img[..., 3] > 0
                val_masks.append(mask)
                z_mask = z[mask]

                val_zs.append(torch.from_numpy(z_mask).to(device))

        all_zs = torch.cat(zs, dim=0)

        if (n_samples is not None) and (all_zs.size(0) > n_samples):
            index = np.array(list(range(all_zs.size(0)))).astype(np.int64)
            np.random.shuffle(index)
            sel_inds = torch.from_numpy(index[:int(n_samples)]).to(device).long()
            x = torch.index_select(all_zs, 0, sel_inds)
        else:
            x = all_zs
        x = x.detach().cpu().numpy()

        cluster = MeanShift(bandwidth=bandwidth, cluster_all=True)
        clustering = cluster.fit(x)
        centers = clustering.cluster_centers_

        np.save(join(dst_scene, 'center.npy'), centers)
        print(centers.shape)

        i = 0
        for val_z, val_m in zip(val_zs, val_masks):
            pred_labels = cluster.predict(val_z)
            im = np.zeros((h, w,)).astype(np.uint8)
            im[val_m] = pred_labels + 1

            filled = np.zeros((h, w, 3)).astype(np.uint8)
            for k, v in color_map.items():
                filled[im == k] = v

            dst_view = join(dst_scene, 'batch{i:09d}'.format(i=i))
            if not os.path.exists(dst_view): os.makedirs(dst_view)
            np.save(join(dst_view, 'labels.npy'), im)
            cv2.imwrite(join(dst_view, 'labels.png'), filled[..., [2, 1, 0]])
            i += 1
