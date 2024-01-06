import numpy as np
from third_party.xiuminglib import xiuminglib as xm
import os
from os.path import join
import json
import cv2
from sklearn.metrics import *
from sklearn.metrics.cluster.supervised import *
from sklearn.metrics.cluster.unsupervised import *

method = 'vq'
n_vals = 8

scenes = ['drums_3072', 'lego_3072', 'hotdog_2163', 'ficus_2188', 'materials_2163']
proj_root = '/home/zhonghongliang/vqnfr_pro_release/decomp/'
pred_root = proj_root + 'nerfvq_nfr3/output/train'
data_root = proj_root + 'data/vis_comps'
label_root = proj_root + 'data/nerf_seg1'
out_root = proj_root + 'results/cluster_abfix.json'

sel_colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
           np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]),
           np.array([127, 0, 0]), np.array([0, 127, 0]), np.array([0, 0, 127]),
           np.array([127, 127, 0]), np.array([127, 0, 127]), np.array([0, 127, 127]),
           np.array([255, 127, 127]), np.array([127, 255, 127]), np.array([127, 127, 255]),
           np.array([255, 255, 127]), np.array([255, 127, 255]), np.array([127, 255, 255]),
           np.array([255, 127, 0]), np.array([255, 0, 127]), np.array([0, 255, 127]),]

embed_c = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
           np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]),
           np.array([128, 0, 0]), np.array([0, 128, 0]), np.array([0, 0, 128]),
           np.array([128, 128, 0]), np.array([128, 0, 128]), np.array([0, 128, 128]),
           np.array([255, 128, 128]), np.array([128, 255, 128]), np.array([128, 128, 255]),
           np.array([255, 255, 128]), np.array([255, 128, 255]), np.array([128, 255, 255]),
           np.array([255, 128, 0]), np.array([255, 0, 128]), np.array([0, 255, 128]),]

def norm_read(img_path, norm=True):
    img = cv2.imread(img_path, -1)
    if len(img.shape)==3 and img.shape[-1]>=3:
        img[..., :3] = img[..., :3][..., [2, 1, 0]] # bgr to rgb
    if norm: return xm.img.normalize_uint(img)
    else: return img


def init_dict():
    metric = {'purity': [], 'f1-micro': [], 'f1-macro': [], 'p-macro': [], 'r-macro': []}
    return metric


def img_embed(arr, colors):
    embed_map = np.zeros(arr.shape[:1] + (1,)).astype(np.int)
    for i in range(len(colors)):
        embed_map[np.all(arr == colors[i], axis=-1)] = np.array([int(i + 1)])
    return np.reshape(embed_map, (-1,))


def purity(coo):
    coo = np.array(coo)
    return np.sum(np.max(coo, axis=0)) / np.sum(coo)


def resort(arr):
    labels = list(set(arr.tolist()))
    labels.sort()
    replaced = np.zeros_like(arr)
    for i in range(len(labels)):
        replaced[np.where(arr == labels[i])] = i
    return replaced


def correspond(gt, pd, replace=True):
    # continuous, start from 0
    gt, pd = resort(gt), resort(pd)
    pd_labels = np.max(pd) + 1

    coo = contingency_matrix(gt, pd)
    coo = np.array(coo)
    label_map = np.argmax(coo, axis=0)

    if replace:
        replaced = np.zeros_like(pd)
        for i in range(pd_labels):
            replaced[np.where(pd == i)] = label_map[i]
    else: replaced = pd
    return coo, label_map, gt, replaced


def process_scene(scene):

    rgba_scene = data_root + '/' + scene
    gt_scene = label_root + '/' + scene
    if method == 'vq':
        pd_scene = pred_root + '/' + scene + '_ref_nfr/lr5e-4/pd_vq/ckpt-5'
    else: pd_scene = pred_root + '/' + scene
    print(rgba_scene, gt_scene, pd_scene)

    gt_imgs, pd_imgs = [], []
    for i in range(n_vals):
        mask = norm_read(join(rgba_scene, 'val_{i:03d}'.format(i=i), 'rgba.png'))[..., -1]
        gt_img = norm_read(join(gt_scene, 'val_{i:03d}'.format(i=i), 'idx.png'), norm=False)
        if method[:2] == 'vq':
            pd_img = norm_read(join(pd_scene, 'batch{i:09d}'.format(i=i), 'embed_map.png'), norm=False)
        else: pd_img = norm_read(join(pd_scene, 'batch{i:09d}'.format(i=i), 'labels.png'), norm=False)

        # only focus on effective pixels
        alpha = mask > 0.8
        gt_imgs.append(gt_img[alpha])
        pd_imgs.append(pd_img[alpha])

    gt = np.concatenate(gt_imgs, axis=0) # n,3
    pd = np.concatenate(pd_imgs, axis=0)

    gt = img_embed(gt, sel_colors)
    pd = img_embed(pd, embed_c)

    coo, _, gt, pd = correspond(gt, pd)

    metric = {}
    metric['purity'] = purity(coo)
    metric['f1-micro'] = f1_score(gt, pd, average='micro')
    metric['f1-macro'] = f1_score(gt, pd, average='macro')
    metric['p-macro'] = precision_score(gt, pd, average='macro')
    metric['r-macro'] = recall_score(gt, pd, average='macro')

    return metric


avg_score = init_dict()
for scene in scenes:
    metric = process_scene(scene)
    for k in avg_score.keys():
        avg_score[k].append(metric[k])

if os.path.exists(out_root):
    with open(out_root, 'r') as f:
        model_score = json.load(f)
else: model_score = {}

model_score[method] = avg_score
with open(out_root, 'w') as f:
    json.dump(model_score, f)

print('\n#---- Average Scores ----#')
for k in avg_score.keys():
    print(k, np.mean(avg_score[k]))

