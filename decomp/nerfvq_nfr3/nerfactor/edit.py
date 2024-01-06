# This file has been modified

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join, basename, exists
from os import makedirs
from absl import app, flags
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
import json
import time

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil, img as imgutil
from third_party.xiuminglib import xiuminglib as xm
from third_party.turbo_colormap import turbo_colormap_data, interpolate_or_clip
import os

logger = logutil.Logger(loggee="test")

def compute_rgb_scales(ckpt):
    """Computes RGB scales that match predicted albedo to ground truth,
    using just the first validation view.
    """
    config_ini = configutil.get_config_ini(ckpt)
    config = ioutil.read_config(config_ini)

    # First validation view
    batch_path = join(config_ini[:-4], 'pd_test', 'ckpt-5')
    batch_glob = os.listdir(batch_path)
    batch_dirs = []
    for batch_dir in  batch_glob:
        if not batch_dir[:5] == 'batch': continue
        batch_dirs.append(join(batch_path, batch_dir))
    data_root = config.get('DEFAULT', 'data_root')

    opt_scale = [[], [], [],]

    for batch_dir in batch_dirs:

        # Find GT path
        view_id = int(batch_dir[-9:])
        view = 'val_{i:03d}'.format(i=view_id)
        scene_name = os.path.basename(os.path.dirname(config_ini))
        vis_root = data_root.replace('nfr_blender', 'vis_comps')

        pred_path = join(batch_dir, 'pred_albedo.png')
        pred_alpha_path = join(batch_dir, 'pred_alpha.png')
        gt_path = join(vis_root, view, 'albedo.png')

        # Load prediction(kd) and GT
        pred = xm.io.img.read(pred_path)
        pred = xm.img.normalize_uint(pred)
        # pred = pred ** 2.2 # undo gamma
        pred_spec = xm.io.img.read(join(batch_dir, 'pred_spec.png'))
        pred_spec = xm.img.normalize_uint(pred_spec)
        pred = pred + pred_spec

        gt = xm.io.img.read(gt_path)  # linear
        gt = xm.img.normalize_uint(gt)

        if scene_name.split('_')[0] in ['drums', 'lego', 'materials', 'chair0', 'kitchen6', 'machine1']:
            print('# Scale With Spec: ', scene_name)
            gt_spec = xm.io.img.read(join(vis_root, view, 'metal.png'))
            gt_spec = xm.img.normalize_uint(gt_spec)
            gt = gt + gt_spec

        gt = xm.img.resize(gt, new_h=pred.shape[0], method='tf')

        rgba = xm.io.img.read(join(data_root, view, 'rgba.png'))
        rgba = xm.img.normalize_uint(rgba)
        rgba = xm.img.resize(rgba, new_h=pred.shape[0], method='tf')
        alpha = rgba[:, :, 3]
        gt = gt[:, :, :3]

        gt = imgutil.linear2srgb(gt)
        pred = imgutil.linear2srgb(pred)

        # Compute color correction scales, in the linear space
        for i in range(3):
            pred_inten = np.sum(pred[:, :, i] * alpha) / np.sum(alpha)
            gt_inten = np.sum(gt[:, :, i] * alpha) / np.sum(alpha)
            opt_scale[i].append(gt_inten / pred_inten)

    opt_scale = np.array(opt_scale)
    print(opt_scale.shape)
    opt_scale = np.mean(opt_scale, axis=-1)
    print(scene_name, opt_scale)
    return opt_scale

def find_vq(vq_root):
    fs = os.listdir(vq_root)
    for f_name in fs:
        if f_name[:5] == 'main_':
            n_vq = int(f_name.split('_')[1])
    return n_vq

def main(_):

    root = '/home/zhonghongliang/vqnfr_pro/nerfvq_nfr3/'
    scene = 'demo_1'
    ckpt = root + 'output/train/'+scene+'_ref_nfr/lr5e-4/checkpoints/ckpt-5'
    img_size = (420, 420)

    # Config
    config_ini = configutil.get_config_ini(ckpt)
    config = ioutil.read_config(config_ini)

    data_type = config.get('DEFAULT', 'data_type')

    # Output directory
    outroot = join(config_ini[:-4], 'pd_edited')

    # Make dataset
    logger.info("Making the actual data pipeline")
    Dataset = datasets.get_dataset_class('ref_nfr')
    dataset = Dataset(config, 'test')
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True, sort=False)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config)
    ioutil.restore_model(model, ckpt)

    if data_type == 'nerf':
        opt_scale = compute_rgb_scales(ckpt)
    else: opt_scale = None

    update_dir = join(config_ini[:-4], 'edited')
    if not os.path.exists(update_dir): os.makedirs(update_dir)
    # For all test views
    logger.info("Running inference")

    ref_out_path = os.path.dirname(config_ini)
    scene_name = basename(ref_out_path)[:-8]  # '_ref_nfr
    vq_out_path = os.path.join(os.path.dirname(ref_out_path), scene_name + '_vq_nfr')
    vq_config_ini = os.path.join(vq_out_path, 'lr5e-4.ini')
    vq_config = ioutil.read_config(vq_config_ini)

    Model = models.get_model_class('vq_nfr')
    vq_ckpt_path = os.path.join(vq_out_path, vq_config_ini[:-4], 'checkpoints', 'ckpt-5')
    vq_model = Model(vq_config)
    ioutil.restore_model(vq_model, vq_ckpt_path)

    print('# Activated, waiting for command...')
    status_path = root + 'status/status.json'
    while True:
        time.sleep(1)

        with open(status_path) as f:
            status_json = json.load(f)
        if status_json['status'] == 'uploaded': # uploaded, finished, waiting, running
            print('# Start Running.')
            status_json['status'] = 'running'
            with open(status_path, 'w') as f:
                json.dump(status_json , f)

            with open(join(update_dir, 'dst.json')) as f:
                dst = json.load(f)
            with open(join(update_dir, 'illum.json')) as f:
                illum = json.load(f)
            dst_env = illum['env']

            for batch_i, batch in enumerate(
                    tqdm(datapipe, desc="Inferring Views", total=n_views)):

                edit_mask_path = join(update_dir, 'batch{i:09d}.npy'.format(i=batch_i))
                edit_mask = np.load(edit_mask_path)
                edit_mask = np.where(edit_mask, 1., 0.)
                edit_mask = cv2.resize(edit_mask, img_size)
                edit_mask = tf.convert_to_tensor(edit_mask, dtype=tf.float32)
                edit_mask = tf.reshape(edit_mask, (-1, 3))

                # Visualize
                outdir_scaled = join(outroot, 'batch{i:09d}'.format(i=batch_i))
                if not exists(outdir_scaled): makedirs(outdir_scaled)

                if dst_env == 'original':
                    _, _, _, to_vis = model.fast_render(
                        batch, mode='test', opt_scale=opt_scale,
                        edit_mask=edit_mask, edit_material=dst)
                    model.vis_batch(to_vis, outdir_scaled, mode='test')

                    _, _, _, to_vis = vq_model.fast_render(
                        batch, mode='test', opt_scale=opt_scale,
                        edit_mask=edit_mask, edit_material=dst,
                        ref_batch=True)
                    vq_model.vis_batch(to_vis, outdir_scaled, mode='test')

                else:
                    _, _, _, to_vis = vq_model.fast_render(
                        batch, mode='test', opt_scale=opt_scale,
                        edit_mask=edit_mask, edit_material=dst,
                        ref_batch=True, dst_env=dst_env)
                    vq_model.vis_batch(to_vis, outdir_scaled, mode='test')

            status_json['status'] = 'finished'
            with open(status_path, 'w') as f:
                json.dump(status_json, f)

            print('# Finish Editing.')



if __name__ == '__main__':
    app.run(main)
