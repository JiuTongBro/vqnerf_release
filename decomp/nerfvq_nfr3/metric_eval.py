import numpy as np
from third_party.xiuminglib import xiuminglib as xm
import os
import json
import tensorflow as tf
import cv2
import sys

eval_rgb = True
eval_kd = True
eval_kskr = True
eval_relight = True

with_rgb = True
with_kd = True
with_kskr = True
with_relight = True

kd_whitebg = False
ks_whitebg = False

k_to_srgb = True
use_scale = True
'''

eval_rgb = False
eval_kd = True
eval_kskr = True
eval_relight = False

with_rgb = False
with_kd = True
with_kskr = True
with_relight = False

kd_whitebg = True
ks_whitebg = True

k_to_srgb = True
use_scale = True
'''
dataset = sys.argv[1]
model_name = sys.argv[2]
print(dataset, model_name)

proj_root = '/home/zhonghongliang/vqnfr_pro_release/decomp/'
# ref_nfr,nvmc,nv,neilf,nfr,pil,base_nfr
'''
if model_name == 'base_nfr':
    pred_root = '/data1/zhl/cmp_results/tvcg/nfr300'
else: pred_root = '/data1/zhl/cmp_results/tvcg/cmps/' + model_name
'''
pred_root = proj_root + 'nerfvq_nfr3/output/train' # output path to our model
out_root = proj_root + 'cmp_results/' + dataset
vis_root = proj_root + 'cmp_results/' + dataset


if model_name == 'base_nfr':
    eval_rgb = False
    eval_relight = False
if model_name in ['neilf', 'pil', 'base_nfr', 'nero']:
    eval_relight = False
if model_name == 'nfr':
    eval_kskr = False

# Our results have been scaled and transfered to sRGB when output
if model_name in ['ref_nfr', 'base_nfr']:
    use_scale = False
    k_to_srgb = False

if dataset == 'nerf':
    scenes = ['drums_3072', 'lego_3072', 'hotdog_2163', 'ficus_2188', 'materials_2163']
    data_root = proj_root + 'data/vis_comps'
    new_h = 512

    with_kskr = False

elif dataset == 'mat':
    scenes = ['chair0_3072', 'kitchen6_7095', 'machine1_3072',]
    data_root = proj_root + 'data/mat_blender'
    new_h = 420

    eval_relight = False
    with_relight = False

elif dataset == 'hw':
    scenes = ['rabbit_-1', 'hwchair_-1', 'toyrabbit_-1'] # nfr failed in redcar
    data_root = proj_root + 'data/1115_hw_data/1115data_1'
    new_h = 420
    use_scale = False

    with_kd = False
    with_kskr = False
    with_relight = False

    k_to_srgb = False

elif dataset == 'dtu':
    scenes = ['dtu_scan24', 'dtu_scan69', 'dtu_scan110', ]
    data_root = proj_root + 'data/dtu_split2'
    new_h = 512
    use_scale = False

    with_kd = False
    with_kskr = False
    with_relight = False

    k_to_srgb = False

elif dataset == 'ours':
    scenes = ['colmap_wshoes', 'colmap_bottle', 'colmap_tools2', ]
    data_root = proj_root + 'data/colmap_split'
    new_h = 420
    use_scale = False

    with_kd = False
    with_kskr = False
    with_relight = False

    k_to_srgb = False

psnr = xm.metric.PSNR('uint8')
ssim = xm.metric.SSIM('uint8')
lpips = xm.metric.LPIPS('uint8')

spec_scenes = ['drums', 'lego', 'materials', 'chair0', 'machine1', 'kitchen6']
env_lights = ['city', 'courtyard', 'forest', 'sunrise', 'night', 'interior', 'studio', 'sunset']
if not dataset in ['nerf', 'mat']:
    env_lights += ['1', '2', '3', '4', '5', '6', '7', '8']

def norm_read(img_path, raw=False):
    img = cv2.imread(img_path, -1)
    if len(img.shape)==3 and img.shape[-1]>=3:
        if raw and img.shape[-1]>3: img = img[..., [2, 1, 0, 3]] # bgr to rgb
        else: img[..., :3] = img[..., :3][..., [2, 1, 0]] # bgr to rgb
    else:
        if len(img.shape)==2:
            img_ = img[..., None]
        elif len(img.shape) == 3 and img.shape[-1] == 1:
            img_ = img
        img = np.repeat(img_, 3, axis=-1)
    return xm.img.normalize_uint(img)


def linear2srgb(np_0to1):
    tensor_0to1 = tf.convert_to_tensor(np_0to1, dtype=tf.float32)

    if isinstance(tensor_0to1, tf.Tensor):
        pow_func = tf.math.pow
        where_func = tf.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb.numpy()


def srgb2linear(np_0to1):
    tensor_0to1 = tf.convert_to_tensor(np_0to1, dtype=tf.float32)

    if isinstance(tensor_0to1, tf.Tensor):
        pow_func = tf.math.pow
        where_func = tf.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.04045
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 / srgb_linear_coeff
    tensor_nonlinear = pow_func((tensor_0to1 + srgb_exponential_coeff - 1) / srgb_exponential_coeff, srgb_exponent)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_rgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_rgb.numpy()


def init_dict(relight=eval_relight, eval_kd=eval_kd):
    metric = {}
    if eval_rgb and with_rgb:
        metric['rgb'] = {'psnr': [], 'ssim': [], 'lpips': [], }
    if eval_kd and with_kd:
        metric['kd'] = {'psnr': [], 'ssim': [], 'lpips': [], }
    if eval_kskr and with_kskr:
        metric['ks'] = {'psnr': [], 'ssim': [], 'lpips': [], }
        metric['rough'] = {'psnr': [], 'ssim': [], 'lpips': [], }
    if eval_relight and with_relight:
        metric['env'] = {'psnr': [], 'ssim': [], 'lpips': [], }

    return metric


def get_scene(pred_root, model_name):
    print(scene, '......')

    if model_name in ['ref_nfr']:
        pd_scene = pred_root + '/' + scene + '_' + model_name + '/lr5e-4/pd_test/ckpt-5'
        relight_dir = pred_root + '/' + scene + '_' + model_name + '/lr5e-4/pd_relit/ckpt-5/'
    elif model_name in ['base_nfr']:
        pd_scene = pred_root + '/' + scene + '_nfr_unit/lr5e-4/vis_z/ckpt-10'
        relight_dir = None
    elif model_name in ['nfr']:
        if dataset in ['ours', 'dtu']:
            pd_scene = pred_root + '/' + scene + '_nerfactor_dtu/lr5e-3/vis_vali/epoch000000100'
            relight_dir = pred_root + '/' + scene + '_nerfactor_dtu/lr5e-3/vis_test/ckpt-10'
        else:
            pd_scene = pred_root + '/' + scene + '_nerfactor/lr5e-3/vis_vali/epoch000000100'
            relight_dir = pred_root + '/' + scene + '_nerfactor/lr5e-3/vis_test/ckpt-10'
    elif model_name in ['nero']:
        scene_prefix = scene.split('_')[0]
        pd_scene = pred_root + '/' + scene_prefix + '_material-val' + '/' + '100000'
        relight_dir = ''
    elif model_name in ['neilf']:
        pd_dir = pred_root + '/' + scene
        pd_scene = pd_dir + '/' + os.listdir(pd_dir)[0] + '/plots/30000'
        relight_dir = ''
    elif model_name in ['nv', 'nvmc']:
        if dataset == 'nerf':
            pd_scene = pred_root + '/' + scene + '/validate'
            relight_dir = pred_root + '/relight/' + scene
        else:
            if scene == 'kitchen6_7095':
                pd_scene = pred_root + '/' + scene + '/dmtet_validate'
                relight_dir = ''
            else:
                pd_scene = pred_root + '/' + scene + '/validate'
                # relight_dir = os.path.dirname(pred_root) + '/relight/' + dataset + '/' + scene
                relight_dir = pred_root + '/relight/' + scene
    elif model_name in ['pil', ]:
        if dataset in ['nerf', 'mat']: dst_folder = 'test_400000'
        else:
            files = os.listdir(pred_root + '/' + scene)
            for f in files:
                if f[:5] == 'test_' and os.path.isdir(pred_root + '/' + scene + '/' + f):
                    dst_folder = f
                    break
        pd_scene = pred_root + '/' + scene + '/' + dst_folder + '/'
        relight_dir = ''
    return pd_scene, relight_dir


def load_albedo(model_name, scene, pd_dir, gt_dir, pd_mask, gt_mask, with_gt=False, opt_scale=None, new_w=None, to_srgb=True):
    # ours&nv&nvmc&neilf&nfr: linear, pil: srgb
    # nv&nvmc&neilf: albedo is kd

    gt_path = gt_dir + 'albedo.png'
    gt_spec_path = gt_dir + 'metal.png'

    if model_name in ['ref_nfr']:
        pd_path = pd_dir + 'pred_basecolor.png'
        pd_spec_path = None
    elif model_name in ['base_nfr']:
        pd_path = pd_dir + 'albedo.png'
        pd_spec_path = pd_dir + 'spec.png'
    elif model_name in ['nfr']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = None
    elif model_name in ['nero']:
        pd_path = pd_dir + 'albedo.jpg'
        pd_spec_path = None
    elif model_name in ['nv', 'nvmc']:
        pd_path = pd_dir + 'pred_kd.png'
        pd_spec_path = None
    elif model_name in ['neilf']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = None
    elif model_name in ['pil']:
        pd_path = pd_dir + 'fine_diffuse.png'
        pd_spec_path = pd_dir + 'fine_specular.png'

    pd = norm_read(pd_path)
    # pd, gt all in linear space:
    if model_name == 'pil': pd = srgb2linear(pd)
    # all use kd for scale:
    if model_name in ['base_nfr', 'pil']:
        pd_spec = norm_read(pd_spec_path)
        if model_name == 'pil': pd += srgb2linear(pd_spec)
        else: pd += pd_spec

    if with_gt:
        gt = norm_read(gt_path)
        if scene.split('_')[0] in spec_scenes:
            gt += norm_read(gt_spec_path)
    else: gt = np.zeros_like(pd)

    if new_h is not None:
        raw_h, raw_w = pd.shape[:2]
        if new_w is None: new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if gt.shape[0] != new_h:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=interp)
        if pd.shape[0] != new_h:
            pd = cv2.resize(pd, (new_w, new_h), interpolation=interp)

    gt = linear2srgb(gt)
    if to_srgb: pd = linear2srgb(pd)
    if opt_scale is not None: pd = pd * opt_scale[None, None, :]

    if kd_whitebg:
        pd = pd[:, :, :3] * pd_mask + (1. - pd_mask)
        gt = gt[:, :, :3] * gt_mask + (1. - gt_mask)
    else:
        pd = pd[:, :, :3] * pd_mask
        gt = gt[:, :, :3] * gt_mask
    pd, gt = np.clip(pd, 0., 1.), np.clip(gt, 0., 1.)

    return pd, gt


def load_spec(model_name, scene, pd_dir, gt_dir, pd_mask, gt_mask, with_gt=False, opt_scale=None, new_w=None, to_srgb=True):
    # ours&nv&nvmc&neilf&nfr: linear, pil: srgb
    # nv&nvmc&neilf: albedo is kd

    gt_path = gt_dir + 'metal.png'

    if model_name in ['ref_nfr']:
        pd_path = pd_dir + 'pred_spec.png'
    elif model_name in ['base_nfr']:
        pd_path = pd_dir + 'spec.png'
    elif model_name in ['nfr']:
        pd_path = None
    elif model_name in ['nero']:
        pd_path = pd_dir + 'albedo.jpg'
        pd_spec_path = pd_dir + 'metallic.jpg'
    elif model_name in ['nv', 'nvmc']:
        pd_path = pd_dir + 'pred_kd.png'
        pd_spec_path = pd_dir + 'pred_ks.png'
    elif model_name in ['neilf']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = pd_dir + 'pred_spec.png'
    elif model_name in ['pil']:
        pd_path = pd_dir + 'fine_specular.png'

    pd = norm_read(pd_path)
    # pd, gt all in linear space:
    if model_name == 'pil': pd = srgb2linear(pd)
    # all use kd for scale:
    if model_name in ['nv', 'nvmc', 'neilf', 'nero']:
        pd_spec = norm_read(pd_spec_path)
        # if model_name == 'pil': pd_spec = srgb2linear(pd_spec)
        pd = pd * pd_spec[..., -1:] # orm

    if with_gt:
        gt = norm_read(gt_path)
        if not scene.split('_')[0] in spec_scenes:
            gt = np.zeros_like(pd)
    else: gt = np.zeros_like(pd)

    if new_h is not None:
        raw_h, raw_w = pd.shape[:2]
        if new_w is None: new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if gt.shape[0] != new_h:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=interp)
        if pd.shape[0] != new_h:
            pd = cv2.resize(pd, (new_w, new_h), interpolation=interp)

    gt = linear2srgb(gt)
    if to_srgb: pd = linear2srgb(pd)
    if opt_scale is not None: pd = pd * opt_scale[None, None, :]

    if ks_whitebg:
        pd = pd[:, :, :3] * pd_mask + (1. - pd_mask)
        gt = gt[:, :, :3] * gt_mask + (1. - gt_mask)
    else:
        pd = pd[:, :, :3] * pd_mask
        gt = gt[:, :, :3] * gt_mask
    pd, gt = np.clip(pd, 0., 1.), np.clip(gt, 0., 1.)

    return pd, gt


def load_rough(model_name, scene, pd_dir, gt_dir, pd_mask, gt_mask, with_gt=False, new_w=None):
    # ours&nv&nvmc&neilf&nfr: linear, pil: srgb
    # nv&nvmc&neilf: albedo is kd

    gt_path = gt_dir + 'rough.png'

    if model_name in ['ref_nfr']:
        pd_path = pd_dir + 'pred_rough.png'
    elif model_name in ['base_nfr']:
        pd_path = pd_dir + 'rough.png'
    elif model_name in ['nfr']:
        pd_path = None
    elif model_name in ['nero']:
        pd_path = pd_dir + 'roughness.jpg'
    elif model_name in ['nv', 'nvmc']:
        pd_path = pd_dir + 'pred_ks.png'
    elif model_name in ['neilf']:
        pd_path = pd_dir + 'pred_rough.png'
    elif model_name in ['pil']:
        pd_path = pd_dir + 'fine_roughness.png'

    pd = norm_read(pd_path)
    if model_name in ['nv', 'nvmc']:
        rough_ = pd[..., 1:2] # orm
        pd = np.repeat(rough_, 3, axis=-1)

    if with_gt: gt = norm_read(gt_path)
    else: gt = np.zeros_like(pd)

    if new_h is not None:
        raw_h, raw_w = pd.shape[:2]
        if new_w is None: new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if gt.shape[0] != new_h:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=interp)
        if pd.shape[0] != new_h:
            pd = cv2.resize(pd, (new_w, new_h), interpolation=interp)

    # black bg in rough
    pd = pd[:, :, :3] * pd_mask
    gt = gt[:, :, :3] * gt_mask
    pd, gt = np.clip(pd, 0., 1.), np.clip(gt, 0., 1.)

    return pd, gt


def load_img(img_path, img_mask=None, new_w=None):

    if img_mask is None:
        img_alpha = norm_read(img_path, raw=True)
        img = img_alpha[..., :3]
        img_mask = img_alpha[..., 3:]
    else: img = norm_read(img_path)

    if new_h is not None:
        raw_h, raw_w = img.shape[:2]
        if new_w is None: new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if img.shape[0] != new_h:
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    img = img[:, :, :3] * img_mask + (1. - img_mask)

    return np.clip(img, 0., 1.)


def load_mask(pd_dir, gt_path):

    gt_rgba = norm_read(gt_path)

    if model_name in ['ref_nfr', 'base_nfr']:
        pd_path = pd_dir + 'pred_alpha.png'
    elif model_name in ['nfr']:
        pd_path = pd_dir + 'gt_alpha.png'
    elif model_name in ['pil']:
        pd_path = pd_dir + 'fine_acc_alpha.png'
    # have no alpha outputs by default, use other outputs to estimate
    elif model_name in ['nv', 'nvmc']:
        pd_path = pd_dir + 'pred_ks.png'
    elif model_name in ['neilf']:
        pd_path = pd_dir + 'pred_albedo.png'

    if model_name in ['nero']:
        pd_path = pd_dir + 'depth.npy'
        pd = np.load(pd_path)
    else:
        pd = norm_read(pd_path)
        # need to reverse the colot if bg is white
        if model_name in ['nv', 'nvmc', 'neilf']: pd = 1. - pd

    if new_h is not None:
        raw_h, raw_w = gt_rgba.shape[:2]
        new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if gt_rgba.shape[0] != new_h:
            gt_rgba = cv2.resize(gt_rgba, (new_w, new_h), interpolation=interp)
        if pd.shape[0] != new_h:
            pd = cv2.resize(pd, (new_w, new_h), interpolation=interp)

    # use stricter thres for GT real-data
    raw_mask = gt_rgba[:, :, 3]
    if dataset in ['nerf', 'mat']:
        gt_mask = np.where(raw_mask > 0.1, 1., 0.)
    else: gt_mask = np.where(raw_mask > 0.95, 1., 0.)

    if model_name in ['nero']:
        d_thres = 10.
        pd_mask = np.where((pd>0.)&(pd<d_thres), 1., 0.)[..., 0]
    else: pd_mask = np.where(np.mean(pd, axis=-1) > 0., 1., 0.)

    return raw_mask[..., None], gt_mask[..., None], pd_mask[..., None], new_w


def _load_kd_for_scale(model_name, scene, pd_dir, gt_dir):
    # ours&nv&nvmc&neilf&nfr: linear, pil: srgb
    # nv&nvmc&neilf: albedo is kd

    gt_path = gt_dir + 'albedo.png'
    rgba_path = gt_dir + 'rgba.png'
    gt_spec_path = gt_dir + 'metal.png'

    if model_name in ['ref_nfr']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = pd_dir + 'pred_spec.png'
    elif model_name in ['base_nfr']:
        pd_path = pd_dir + 'albedo.png'
        pd_spec_path = pd_dir + 'spec.png'
    elif model_name in ['nfr']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = None
    elif model_name in ['nero']:
        pd_path = pd_dir + 'albedo.jpg'
        pd_spec_path = None
    elif model_name in ['nv', 'nvmc']:
        pd_path = pd_dir + 'pred_kd.png'
        pd_spec_path = None
    elif model_name in ['neilf']:
        pd_path = pd_dir + 'pred_albedo.png'
        pd_spec_path = None
    elif model_name in ['pil']:
        pd_path = pd_dir + 'fine_diffuse.png'
        pd_spec_path = pd_dir + 'fine_specular.png'

    pd = norm_read(pd_path)
    gt = norm_read(gt_path)

    # pd, gt all in linear space:
    if model_name == 'pil': pd = srgb2linear(pd)

    # all use kd for scale:
    if model_name in ['ref_nfr', 'base_nfr', 'pil']:
        pd_spec = norm_read(pd_spec_path)
        if model_name == 'pil': pd += srgb2linear(pd_spec)
        else: pd += pd_spec

    if scene.split('_')[0] in spec_scenes:
        gt += norm_read(gt_spec_path)

    gt_rgba = norm_read(rgba_path)

    if new_h is not None:
        raw_h, raw_w = gt_rgba.shape[:2]
        new_w = int(new_h * raw_w / raw_h)
        interp = cv2.INTER_LINEAR if new_h > raw_h else cv2.INTER_AREA
        if gt.shape[0] != new_h:
            gt = cv2.resize(gt, (new_w, new_h), interpolation=interp)
        if gt_rgba.shape[0] != new_h:
            gt_rgba = cv2.resize(gt_rgba, (new_w, new_h), interpolation=interp)
        if pd.shape[0] != new_h:
            pd = cv2.resize(pd, (new_w, new_h), interpolation=interp)

    mask = gt_rgba[:, :, 3]
    pd, gt = linear2srgb(pd[..., :3]), linear2srgb(gt[..., :3])
    return pd, gt, mask


def compute_rgb_scales(model_name, scene, views, pd_scene, gt_scene):
    opt_scale = [[], [], [], ]

    for view_ in views:
        if model_name in ['pil', 'base_nfr']:
            if not view_[:3] == 'val': continue
        elif (not view_[:5] == 'batch') and (not model_name == 'nero'):
            continue

        if model_name == 'nero':
            view = 'batch{i:09d}'.format(i=int(view_))
        else: view = view_

        pd_view = pd_scene + '/' + view_ + '/'
        rgba_view = gt_scene + '/val_' + view[-3:] + '/'

        pred, gt, alpha = _load_kd_for_scale(model_name, scene, pd_view, rgba_view)

        for i in range(3):
            pred_inten = np.sum(pred[:, :, i] * alpha) / np.sum(alpha)
            gt_inten = np.sum(gt[:, :, i] * alpha) / np.sum(alpha)
            opt_scale[i].append(gt_inten / pred_inten)

    opt_scale = np.array(opt_scale)
    print(opt_scale.shape)
    opt_scale = np.mean(opt_scale, axis=-1)
    print(scene, opt_scale)
    return opt_scale


def write_img(out_path, img):
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(out_path, img[..., [2, 1, 0]])


def process_scene(scene):
    metric = init_dict()

    pd_scene, relight_dir = get_scene(pred_root, model_name)

    gt_scene = data_root + '/' + scene
    print(pd_scene, gt_scene)
    views = os.listdir(pd_scene)

    if use_scale:
        print('Scale kd...')
        opt_scale = compute_rgb_scales(model_name, scene, views, pd_scene, gt_scene)
    else: opt_scale = None

    for view_ in views:
        if model_name in ['pil', 'base_nfr']:
            if not view_[:3] == 'val': continue
        elif (not view_[:5] == 'batch') and (not model_name == 'nero'):
            continue

        if model_name == 'nero':
            view = 'batch{i:09d}'.format(i=int(view_))
        else: view = view_

        vis_dir = os.path.join(vis_root, model_name, scene, view)
        if not os.path.exists(vis_dir): os.makedirs(vis_dir)

        gt_view_dir = gt_scene + '/val_' + view[-3:] + '/'
        rgba_view = gt_view_dir + 'rgba.png'

        pd_view_dir = pd_scene + '/' + view_ + '/'
        raw_mask, gt_mask, pd_mask, new_w = load_mask(pd_view_dir, rgba_view)

        if eval_rgb:
            if model_name[:2] == 'nv':
                pd_view = pd_view_dir + 'pred_opt.png'
            elif model_name == 'neus':
                pd_view = pd_view_dir + 'rgb.png'
            elif model_name == 'pil':
                pd_view = pd_view_dir + 'fine_rgb.png'
            elif model_name == 'nero':
                pd_view = pd_view_dir + 'rgb_pr.jpg'
            else:
                pd_view = pd_view_dir + 'pred_rgb.png'

            pd = load_img(pd_view, pd_mask, new_w=new_w)
            write_img(os.path.join(vis_dir, 'pred_rgb.png'), pd)

            if with_rgb:
                gt = load_img(rgba_view, gt_mask, new_w=new_w)
                write_img(os.path.join(vis_dir, 'gt_rgb.png'), gt)

                # Align, following NeRFactor
                pd = alpha_blend(pd, raw_mask)
                gt = alpha_blend(gt, raw_mask)

                metric['rgb']['psnr'].append(psnr(gt, pd).tolist())
                metric['rgb']['ssim'].append(ssim(gt, pd).tolist())
                metric['rgb']['lpips'].append(lpips(gt, pd).tolist())

        if eval_kd:

            pd, gt = load_albedo(model_name, scene, pd_view_dir, gt_view_dir, pd_mask, gt_mask,
                                 with_gt=with_kd, opt_scale=opt_scale, new_w=new_w, to_srgb=k_to_srgb)
            write_img(os.path.join(vis_dir, 'pred_kd.png'), pd)

            if with_kd:
                write_img(os.path.join(vis_dir, 'gt_kd.png'), gt)

                pd = alpha_blend(pd, raw_mask)
                gt = alpha_blend(gt, raw_mask)

                metric['kd']['psnr'].append(psnr(gt, pd).tolist())
                metric['kd']['ssim'].append(ssim(gt, pd).tolist())
                metric['kd']['lpips'].append(lpips(gt, pd).tolist())

        if eval_kskr:
            pd, gt = load_spec(model_name, scene, pd_view_dir, gt_view_dir, pd_mask, gt_mask,
                               with_gt=with_kskr, opt_scale=opt_scale, new_w=new_w, to_srgb=k_to_srgb)
            write_img(os.path.join(vis_dir, 'pred_metal.png'), pd)

            if with_kskr:
                write_img(os.path.join(vis_dir, 'gt_metal.png'), gt)

                pd = alpha_blend(pd, raw_mask)
                gt = alpha_blend(gt, raw_mask)

                metric['ks']['psnr'].append(psnr(gt, pd).tolist())
                metric['ks']['ssim'].append(ssim(gt, pd).tolist())
                metric['ks']['lpips'].append(lpips(gt, pd).tolist())

            pd, gt = load_rough(model_name, scene, pd_view_dir, gt_view_dir, pd_mask, gt_mask,
                                with_gt=with_kskr, new_w=new_w)
                                # with_gt = False, new_w = new_w)
            write_img(os.path.join(vis_dir, 'pred_rough.png'), pd)

            if with_kskr:
                write_img(os.path.join(vis_dir, 'gt_rough.png'), gt)

                pd = alpha_blend(pd, raw_mask)
                gt = alpha_blend(gt, raw_mask)

                metric['rough']['psnr'].append(psnr(gt, pd).tolist())
                metric['rough']['ssim'].append(ssim(gt, pd).tolist())
                metric['rough']['lpips'].append(lpips(gt, pd).tolist())

        # Relight
        if eval_relight and (not ((model_name in ['nv', 'nvmc']) and (scene == 'toyrabbit_-1'))):
            for env_l in env_lights:
                if model_name in ['nv', 'nvmc']:
                    pd_view = relight_dir + '/val_' + view[-3:] + '/rgba_' + env_l + '.png'
                    pd = load_img(pd_view, new_w=new_w)
                else:
                    pd_view = relight_dir + '/' + view + '/' + 'pred_rgb_probes_' + env_l + '.png'
                    pd = load_img(pd_view, pd_mask, new_w=new_w)
                write_img(os.path.join(vis_dir, 'pred_' + env_l + '.png'), pd)

                if with_relight:
                    gt_view = gt_scene + '/val_' + view[-3:] + '/rgba_' + env_l + '.png'
                    gt = load_img(gt_view, gt_mask, new_w=new_w)
                    write_img(os.path.join(vis_dir, 'gt_' + env_l + '.png'), gt)

                    pd = alpha_blend(pd, raw_mask)
                    gt = alpha_blend(gt, raw_mask)

                    metric['env']['psnr'].append(psnr(gt, pd).tolist())
                    metric['env']['ssim'].append(ssim(gt, pd).tolist())
                    metric['env']['lpips'].append(lpips(gt, pd).tolist())

    return metric


# Since different alpha_thres used in different model can easily affect the final results (boundary differences)
# We follow the NeRFactor to set a 'standard' bg for all results, focusing on the effective pixels
def alpha_blend(img, alpha):
    # stricter for real-data
    if dataset in ['nerf' 'mat']: alpha_thres = 0.8
    else: alpha_thres = 0.95

    mask = np.where(alpha>alpha_thres, 1., 0.)
    img = img * mask + (1. - mask)
    img = np.clip(img, 0., 1.)
    img = (img * 255).astype(np.uint8)
    return img


def compute_mean(model_score, pref):
    _psnr = np.mean(model_score[scene][pref]['psnr'])
    _ssim = np.mean(model_score[scene][pref]['ssim'])
    _lpips = np.mean(model_score[scene][pref]['lpips'])
    return _psnr, _ssim, _lpips


model_score = {}
avg_score = init_dict()
for scene in scenes:

    metric = process_scene(scene)
    model_score[scene] = metric

    # Avg RGB
    if eval_rgb and with_rgb:
        pref = 'rgb'
        psnr_, ssim_, lpips_ = compute_mean(model_score, pref)
        avg_score[pref]['psnr'].append(psnr_)
        avg_score[pref]['ssim'].append(ssim_)
        avg_score[pref]['lpips'].append(lpips_)

    if not dataset in ['nerf', 'mat']: continue

    if eval_kd and with_kd:
        pref = 'kd'
        psnr_, ssim_, lpips_ = compute_mean(model_score, pref)
        avg_score[pref]['psnr'].append(psnr_)
        avg_score[pref]['ssim'].append(ssim_)
        avg_score[pref]['lpips'].append(lpips_)

    if eval_kskr and with_kskr:
        pref = 'ks'
        psnr_, ssim_, lpips_ = compute_mean(model_score, pref)
        avg_score[pref]['psnr'].append(psnr_)
        avg_score[pref]['ssim'].append(ssim_)
        avg_score[pref]['lpips'].append(lpips_)

        pref = 'rough'
        psnr_, ssim_, lpips_ = compute_mean(model_score, pref)
        avg_score[pref]['psnr'].append(psnr_)
        avg_score[pref]['ssim'].append(ssim_)
        avg_score[pref]['lpips'].append(lpips_)

    if eval_relight and with_relight:
        pref = 'env'
        psnr_, ssim_, lpips_ = compute_mean(model_score, pref)
        avg_score[pref]['psnr'].append(psnr_)
        avg_score[pref]['ssim'].append(ssim_)
        avg_score[pref]['lpips'].append(lpips_)

if not os.path.exists(out_root): os.makedirs(out_root)
with open(out_root + '/' + model_name + '.json', 'w') as f:
    json.dump(model_score, f)

eval_list = []
if eval_rgb and with_rgb: eval_list.append('rgb')
if eval_kd and with_kd: eval_list.append('kd')
if eval_kskr and with_kskr:
    eval_list.append('ks')
    eval_list.append('rough')
if eval_relight and with_relight: eval_list.append('env')


for pref in eval_list:
    print('\n#---- Average ' + pref + ' Scores ----#')
    print('#PSNR: ', np.mean(avg_score[pref]['psnr']))
    print('#SSIM: ', np.mean(avg_score[pref]['ssim']))
    print('#LPIPS: ', np.mean(avg_score[pref]['lpips']))

