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

"""A general training and validation pipeline.
"""

from os.path import join, dirname, exists
from shutil import rmtree
from time import time
from collections import deque
from tqdm import tqdm
from absl import app, flags
import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from third_party.xiuminglib import xiuminglib as xm
from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil

import torch
from nerfactor.util.torch_kmeans import kmeans

flags.DEFINE_string(
    'config', 'gnerf.ini', "base .ini file in config/ or a full path")
flags.DEFINE_string('config_override', '', "e.g., 'key1=value1,key2=value2'")
flags.DEFINE_boolean('debug', False, "debug mode switch")
flags.DEFINE_enum(
    'device', 'gpu', ['cpu', 'gpu'], "running on what type of device(s)")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="trainvali")


def main(_):
    if FLAGS.debug: logger.warn("Debug mode: on")

    # Configurations
    config_ini = FLAGS.config
    if not exists(config_ini):
        config_ini = join(dirname(__file__), 'config', FLAGS.config)
    config = ioutil.read_config(config_ini)
    # Any override?
    if FLAGS.config_override != '':
        for kv in FLAGS.config_override.split(','):
            k, v = kv.split('=')
            config.set('DEFAULT', k, v)

    # Output directory
    config_dict = configutil.config2dict(config)
    xname = config.get('DEFAULT', 'xname').format(**config_dict)
    outroot = config.get('DEFAULT', 'outroot')
    outdir = join(outroot, xname)
    overwrite = config.getboolean('DEFAULT', 'overwrite')
    ioutil.prepare_outdir(outdir, overwrite=overwrite)
    logger.info("For results, see:\n\t%s", outdir)

    # Dump actual configuration used to disk
    config_out = outdir.rstrip('/') + '.ini'
    ioutil.write_config(config, config_out)

    _seed = config.getint('DEFAULT', 'random_seed')
    print('#* Random Seed: ', _seed)
    tf.random.set_seed(_seed)
    np.random.seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Make training dataset
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset_train = Dataset(config, 'train', debug=FLAGS.debug)
    global_bs_train = dataset_train.bs  # batch_size
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    data_type = config.get('DEFAULT', 'data_type')

    # full-img training
    datapipe_train = dataset_train.build_pipeline(no_batch=no_batch, no_shuffle=True)
    total_sample_vq = config.getint('DEFAULT', 'total_sample_vq')
    per_sample_vq = total_sample_vq // dataset_train.get_n_views()
    vq_test_batch = prepare_vq_data(config, per_sample_vq, datapipe_train, data_type)

    # Make validation dataset
    dataset_vali = Dataset(config, 'vali', debug=FLAGS.debug)
    global_bs_vali = dataset_vali.bs  # maybe different from training
    try:
        datapipe_vali = dataset_vali.build_pipeline(no_batch=no_batch)
    except FileNotFoundError:
        datapipe_vali = None
    # Sample validation batches, and just stick with them
    if datapipe_vali is None:
        vali_batches = None
    else:
        n_vali_batches = config.getint('DEFAULT', 'vali_batches')
        vali_batches = datapipe_vali.take(n_vali_batches)

    # Model
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    model.register_trainable()

    # Optimizer
    lr = config.getfloat('DEFAULT', 'lr')
    lr_decay_steps = config.getint('DEFAULT', 'lr_decay_steps', fallback=-1)
    if lr_decay_steps > 0:
        lr_decay_rate = config.getfloat('DEFAULT', 'lr_decay_rate')
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            lr, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate)

    kwargs = {'learning_rate': lr, 'amsgrad': True}
    clipnorm = config.getfloat('DEFAULT', 'clipnorm')
    clipvalue = config.getfloat('DEFAULT', 'clipvalue')
    err_msg = \
        "Both `clipnorm` and `clipvalue` are active -- turn one off"
    if clipnorm > 0:
        assert clipvalue < 0, err_msg
        kwargs['clipnorm'] = clipnorm
    if clipvalue > 0:
        assert clipnorm < 0, err_msg
        kwargs['clipvalue'] = clipvalue
    optimizer = tf.keras.optimizers.Adam(**kwargs)

    # kwargs = {'learning_rate': lr}
    # optimizer = tf.keras.optimizers.SGD(**kwargs)

    # Resume from checkpoint, if any
    ckptdir = join(outdir, 'checkpoints')
    assert model.trainable_registered, (
        "Register the trainable layers to have them tracked by the "
        "checkpoint")
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optimizer, net=model)
    keep_recent_epochs = config.getint('DEFAULT', 'keep_recent_epochs')
    if keep_recent_epochs <= 0:
        keep_recent_epochs = None  # keep all epochs
    ckptmanager = tf.train.CheckpointManager(
        ckpt, ckptdir, max_to_keep=keep_recent_epochs)
    # training from a ckpt can result in warnings about unused variables in the old optimizer, can be ignored
    ckpt.restore(ckptmanager.latest_checkpoint)
    if ckptmanager.latest_checkpoint:
        logger.info(
            "Resumed from step:\n\t%s", ckptmanager.latest_checkpoint)
    else:
        logger.info("Started from scratch")

    # Summary directories
    writer_train = tf.summary.create_file_writer(
        join(outdir, 'summary_train'))
    writer_vali = tf.summary.create_file_writer(
        join(outdir, 'summary_vali'))
    vali_vis_epoch_dir = join(outdir, 'vis_vali', 'epoch{e:09d}')
    full_vis_epoch_dir = join(outdir, 'vis_vali', 'vis_params', 'epoch{e:09d}')

    vali_vis_batch_rawf = join(
        vali_vis_epoch_dir, 'batch{b:09d}_raw.pickle')
    vali_vis_thres_dir = join(vali_vis_epoch_dir, '%s')
    vali_vis_batch_dir = join(vali_vis_thres_dir, 'batch{b:09d}')

    init_z_root = join(dirname(dirname(outroot)), 'cluster')
    # init_z_root = join(dirname(dirname(outroot)), 'ablation_cluster')
    if not exists(init_z_root): os.makedirs(init_z_root)
    exp_split = os.path.basename(outroot).split('_')
    scene_name = exp_split[0] + '_' + exp_split[1]
    init_z_path = join(init_z_root, scene_name + '.npy')

    vis_view = config.getint('DEFAULT', 'vis_view')

    num_embed = config.getint('DEFAULT', 'num_embed')
    num_drop = config.getint('DEFAULT', 'num_drop')
    thres_str = config.get('DEFAULT', 'thres_str')
    makeups = [0.] * (num_embed - num_drop)
    if not thres_str == '-':
        train_thres = makeups + [float(x) for x in thres_str.split(';')]
    else:
        train_thres = makeups
    x_list = list(range(num_embed - num_drop, num_embed + 1))
    train_thres = np.array(train_thres)
    print('# Train Thres: ', train_thres)
    val_thres_list = [np.array([0.] * (num_embed - i) + [1.] * i)
                      for i in range(num_drop + 1)]
    val_thres_list.reverse()  # raw val_thres is reversed, the last with the less material
    print('# Val Thres: ', val_thres_list[0], val_thres_list[-1])

    # extension yourself in your overriding function (this makes the
    # pipeline general and not specific to any model)

    # ====== Training loop ======
    epochs = config.getint('DEFAULT', 'epochs')
    ckpt_period = config.getint('DEFAULT', 'ckpt_period')
    vali_period = config.getint('DEFAULT', 'vali_period')
    step_restored = ckpt.step.numpy()

    if step_restored == 0:
        num_embed = config.getint('DEFAULT', 'num_embed')
        init_batch_vis = []
        for batch_i, _batch in enumerate(datapipe_train):
            # Validate on this validation batch
            batch = outer_sample(_batch, config, data_type)

            if data_type == 'nerf':
                id_, hw, _, _, _, alpha, pred_alpha, xyz, _, _ = batch
            else:
                id_, hw, _, _, _, alpha, pred_alpha, xyz, _ = batch
            mask = alpha[:, 0] > 0
            xyz = tf.boolean_mask(xyz, mask)
            if batch_i == 0: print('First Train Pipe Elem: ', tf.shape(xyz), xyz[0])

            to_vis = model.init_z(batch)
            init_batch_vis.append(to_vis['z_pred'].numpy())
        z_cluster(model, init_batch_vis, init_z_path, num_embed, seed=_seed)
    else:
        print('# Skip Init Z...')

    for _ in tqdm(range(step_restored, epochs), desc="Training epochs"):
        # ------ Train on all batches ------
        batch_loss, batch_vis, batch_time = [], [], []
        train_step = ckpt.step.numpy()

        loss_dicts = []
        for batch_i, _batch in enumerate(datapipe_train):
            t0 = time()
            batch = outer_sample(_batch, config, data_type)
            loss, to_vis, loss_dict = \
                train_iter(model, batch, optimizer, global_bs_train, train_thres)

            batch_time.append(time() - t0)
            batch_loss.append(loss)
            loss_dicts.append(loss_dict)

            if FLAGS.debug:
                logger.warn(
                    "Debug mode: skipping the rest of this epoch")
                break

        assert batch_time, "Dataset is empty"

        # Record step
        ckpt.step.assign_add(1)
        step = ckpt.step.numpy()

        # Checkpoint and summarize/visualize training
        if step % ckpt_period == 0:
            # Save checkpoint
            saved_path = ckptmanager.save()
            logger.info("Checkpointed step %s:\n\t%s", step, saved_path)
            # Summarize training
            with writer_train.as_default():
                tf.summary.scalar(
                    "loss_train", tf.reduce_mean(batch_loss), step=step)
                tf.summary.scalar(
                    "batch_time_train", tf.reduce_mean(batch_time),
                    step=step)

        # ------ Validation ------
        # --- save loss

        if vali_batches is not None and vali_period > 0 and step % vali_period == 0:

            ckpt_step = step
            vis_dirs = []

            losses = loss_dicts[0]
            for k, v in losses.items():
                losses[k] = tf.reduce_mean(v).numpy()
            for i in range(1, len(loss_dicts)):
                for k, v in loss_dicts[i].items():
                    losses[k] += tf.reduce_mean(v).numpy()
            for k, v in losses.items():
                losses[k] = v.tolist()
            loss_dir = vali_vis_epoch_dir.format(e=ckpt_step)
            if not os.path.exists(loss_dir): os.makedirs(loss_dir)
            with open(join(loss_dir, 'loss.json'), 'w') as f:
                json.dump(losses, f)

            # --- save vq error
            vq_test_scores = {'vqrgb': [], 'chromaticity': []}
            for val_thres in val_thres_list:
                # Validate on this validation batch
                '''
                for batch_i, batch in enumerate(vali_batches):
                    if not batch_i == vis_view: continue
                    loss_dict = vali_vq(model, batch, val_thres)
                '''
                loss_dict = vali_vq(model, vq_test_batch, val_thres)
                vq_test_scores['vqrgb'].append(tf.reduce_mean(loss_dict['vqrgb']).numpy().tolist())
                if 'chromaticity' in loss_dict.keys():
                    vq_test_scores['chromaticity'].append(tf.reduce_mean(loss_dict['chromaticity']).numpy().tolist())
                else: vq_test_scores['chromaticity'].append(tf.reduce_mean(loss_dict['vqrgb']).numpy().tolist())
            with open(join(loss_dir, 'vq_test_loss.json'), 'w') as f:
                json.dump(vq_test_scores, f)

            drop_losses = []
            for i in range(len(val_thres_list)):
                # drop_losses.append(vq_test_scores['vqrgb'][i] + vq_test_scores['chromaticity'][i])
                drop_losses.append(vq_test_scores['chromaticity'][i])
            drop_losses = np.array(drop_losses)

            plt.clf()
            plt.plot(x_list, drop_losses)
            plt.savefig(join(loss_dir, 'vq_num.png'))

            best_thres = config.getfloat('DEFAULT', 'best_thres')
            main_thres = best_thres

            for i in range(1, len(val_thres_list) - 1):
                if drop_losses[i - 1] > drop_losses[i]:
                    best_flag = True
                    for j in range(i + 1, len(val_thres_list)):
                        if drop_losses[i] - drop_losses[j] > main_thres:
                            best_flag = False
                            break
                else:
                    best_flag = False
                if best_flag:
                    main_vq = i
                    break
            if not best_flag: main_vq = len(val_thres_list) - 1

            # --- save val results
            for i in range(len(val_thres_list)):
                # Run validation on all validation batches
                val_thres = val_thres_list[i]
                batch_loss, batch_vis = [], []
                for batch_i, batch in enumerate(vali_batches):
                    # Validate on this validation batch
                    loss, to_vis, _ = vali_iter(model, batch, global_bs_vali, val_thres,
                                                full_vis=(batch_i == vis_view))

                    batch_loss.append(loss)
                    batch_vis.append(to_vis)

                # Summarize/visualize validation
                with writer_vali.as_default():
                    tf.summary.scalar(
                        "loss_vali", tf.reduce_mean(batch_loss), step=ckpt_step)
                    for batch_i, to_vis in enumerate(batch_vis):
                        raw_f = vali_vis_batch_rawf.format(e=ckpt_step, b=batch_i)
                        # vis_dir = (vali_vis_batch_dir % str(num_embed - num_drop + i)).format(e=ckpt_step, b=batch_i)

                        # Generate Afterwards
                        if i == main_vq:
                            vis_dir = (vali_vis_batch_dir % ('main_' + str(num_embed - num_drop + i))).format(
                                e=ckpt_step, b=batch_i)
                        else:
                            vis_dir = (vali_vis_batch_dir % str(num_embed - num_drop + i)).format(e=ckpt_step,
                                                                                                  b=batch_i)

                        # only when all code involves
                        if (batch_i == vis_view) and (i == len(val_thres_list) - 1):
                            full_vis_path = full_vis_epoch_dir.format(e=ckpt_step)
                            if not os.path.exists(full_vis_path): os.makedirs(full_vis_path)
                        else:
                            full_vis_path = None
                        model.vis_batch(to_vis, vis_dir, mode='vali',
                                        dump_raw_to=raw_f, simp=True,
                                        full_vis_path=full_vis_path)
                        vis_dirs.append(vis_dir)

    save_metas(outdir)


# outer sample to keep deterministic
def outer_sample(batch, config, data_type, alpha_thres=0.9):
    bs = config.getint('DEFAULT', 'n_rays_per_step')

    if data_type == 'nerf':
        id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
    else:
        id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
        lvis = None

    img_h, img_w = tuple(hw[0, :].numpy())
    id_ = tf.reshape(id_, (img_h, img_w,))
    hw = tf.reshape(hw, (img_h, img_w, -1))
    rayo = tf.reshape(rayo, (img_h, img_w, -1))
    rayd = tf.reshape(rayd, (img_h, img_w, -1))
    rgb = tf.reshape(rgb, (img_h, img_w, -1))
    alpha = tf.reshape(alpha, (img_h, img_w,))
    pred_alpha = tf.reshape(pred_alpha, (img_h, img_w,))
    xyz = tf.reshape(xyz, (img_h, img_w, -1))
    normal = tf.reshape(normal, (img_h, img_w, -1))
    if data_type == 'nerf':
        lvis = tf.reshape(lvis, (img_h, img_w, -1))

    jitters = tf.constant([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                           [0, 1], [1, -1], [1, 0], [1, 1]], dtype=tf.int32)  # n,2

    # Training: sample rays
    size1, size2 = tf.shape(rgb)[0], tf.shape(rgb)[1]
    n_jitters = tf.shape(jitters)[0]
    coords = tf.stack(tf.meshgrid(tf.range(1, size1 - 1), tf.range(1, size2 - 1), indexing='ij'), axis=-1)

    coords_jitters = jitters[:, None, None, :] + coords[None, ...]  # n,h,w,2
    coords_jitters = tf.concat(coords_jitters, axis=0)
    rgb_jitters = tf.gather_nd(rgb, coords_jitters)

    # jitter_inds = tf.argmax(tf.reduce_max(tf.abs(rgb_jitters - rgb[None, 1:-1, 1:-1, :]), axis=-1), axis=0)
    jitter_inds = tf.random.uniform(
        (size1 - 2, size2 - 2), minval=0, maxval=n_jitters, dtype=tf.int32)
    coords_jitters = tf.reshape(tf.transpose(coords_jitters, [1, 2, 0, 3]), (-1, n_jitters, 2))
    coords_jitter = tf.gather_nd(coords_jitters, tf.reshape(jitter_inds, (-1, 1)), batch_dims=1)
    coords_jitter = tf.reshape(coords_jitter, (size1 - 2, size2 - 2, 2))

    alpha_jitter = tf.gather_nd(alpha, tf.reshape(coords_jitter, (-1, 2)))
    alpha_jitter = tf.reshape(alpha_jitter, (size1 - 2, size2 - 2))

    # Keep only the foreground coordinates?
    if alpha_thres is None:
        coords = tf.reshape(coords, (-1, 2))
        coords_jitter = tf.reshape(coords_jitter, (-1, 2))
    else:
        alpha.set_shape((None, None))  # required by graph mode
        alpha_jitter.set_shape((None, None))  # required by graph mode
        mask = (alpha[1:-1, 1:-1] > alpha_thres) & (alpha_jitter > alpha_thres)

        coords = tf.boolean_mask(coords, mask)  # n,2
        coords_jitter = tf.boolean_mask(coords_jitter, mask)  # n,2

    # Use tf.random instead of np.random here so that the randomness is
    # correct even if we compile this to static graph using tf.function

    # np_inds = np.random.uniform(low=0, high=tf.shape(coords)[0], size=(self.bs,))
    # select_ind = tf.convert_to_tensor(np_inds, dtype=tf.int32)
    select_ind = tf.random.uniform(
        (bs,), minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
    # print('#--- Outer Sample First Ind: ', select_ind[0])

    select_coords = tf.gather_nd(coords, select_ind[:, None])  # n,2
    select_jitter = tf.gather_nd(coords_jitter, select_ind[:, None])  # n,2
    select_ind = tf.concat([select_coords, select_jitter], axis=-1)
    select_ind = tf.reshape(select_ind, (-1, 2))  # [p1, p1_n, p2, p2_n, ...]

    id_ = tf.gather_nd(id_, select_ind)
    hw = tf.gather_nd(hw, select_ind)
    rayo = tf.gather_nd(rayo, select_ind)
    rayd = tf.gather_nd(rayd, select_ind)
    rgb = tf.gather_nd(rgb, select_ind)
    alpha = tf.gather_nd(alpha, select_ind)
    alpha = tf.reshape(alpha, (-1, 1))
    pred_alpha = tf.gather_nd(pred_alpha, select_ind)
    pred_alpha = tf.reshape(pred_alpha, (-1, 1))
    xyz = tf.gather_nd(xyz, select_ind)
    normal = tf.gather_nd(normal, select_ind)

    if data_type == 'nerf':
        if lvis is None: print('# NeRF Data Requires lvis!')
        lvis = tf.gather_nd(lvis, select_ind)
        return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis
    else:
        return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal


# just to locate the centers within the latent distribution, no correspondence to specified materials
def z_cluster(model, init_batch_vis, init_z_path, num_embed, device='cpu', n_samples=None, seed=1):
    init_zs = np.concatenate(init_batch_vis, axis=0)
    if n_samples is None:
        x = torch.from_numpy(init_zs).to(device)
    else:
        index = np.array(range(init_zs.shape[0]))
        np.random.shuffle(index)
        sel_inds = index[:int(n_samples)]
        x = torch.from_numpy(init_zs[sel_inds]).to(device)

    print('# X first elem: ', x.size(), x[0][0])
    _, cluster_centers = kmeans(
        X=x, num_clusters=num_embed, distance='euclidean', device=device, seed=seed)
    z_centers = cluster_centers.detach().cpu().numpy()

    print('# init centers shape: ', z_centers.shape)
    print('# Z Center First: ', z_centers[0][0])
    np.save(init_z_path, z_centers)


def save_metas(outdir):
    outdir = join(outdir, 'vis_vali')
    metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'psnr_luma': [], 'ssim_luma': [], 'mse': []}
    for e_dir in os.listdir(outdir):
        if not e_dir[:5] == 'epoch': continue
        epoch_metric = {'psnr': [], 'ssim': [], 'lpips': [], 'psnr_luma': [], 'ssim_luma': [], 'mse': []}
        dirs = os.listdir(join(outdir, e_dir))
        for dir in dirs:
            if not dir[:5] == 'batch': continue
            json_path = join(outdir, e_dir, dir, 'metadata.json')
            with open(json_path) as f:
                js = json.load(f)
            for k, v in js.items():
                if k in epoch_metric:
                    epoch_metric[k].append(v)
        for k in metrics.keys():
            metrics[k].append(np.mean(epoch_metric[k]))
    out_meta = join(outdir, 'metas.json')
    with open(out_meta, 'w') as f:
        json.dump(metrics, f)


def prepare_vq_data(config, per_sample_n, datapipe_train, data_type):
    ids, hws, rayos, rayds, rgbs, alphas, pred_alphas, xyzs, normals = \
        [], [], [], [], [], [], [], [], [],
    if data_type == 'nerf': lvises = []
    for batch_i, _batch in enumerate(datapipe_train):
        batch = outer_sample(_batch, config, data_type)
        if data_type == 'nerf':
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
        else:
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch

        mask = alpha[:, 0] > 0
        masked_id = tf.boolean_mask(id_, mask)

        select_ind = tf.random.uniform(
            (per_sample_n,), minval=0, maxval=tf.shape(masked_id)[0], dtype=tf.int32)
        if batch_i == 0: print('# ---- VQ Test Sample First Random Ind: ', select_ind[0])

        ids.append(sample_tensor(id_, mask, select_ind))
        hws.append(sample_tensor(hw, mask, select_ind))
        rayos.append(sample_tensor(rayo, mask, select_ind))
        rayds.append(sample_tensor(rayd, mask, select_ind))
        rgbs.append(sample_tensor(rgb, mask, select_ind))
        alphas.append(sample_tensor(alpha, mask, select_ind))
        pred_alphas.append(sample_tensor(pred_alpha, mask, select_ind))
        xyzs.append(sample_tensor(xyz, mask, select_ind))
        normals.append(sample_tensor(normal, mask, select_ind))
        if data_type == 'nerf':
            lvises.append(sample_tensor(lvis, mask, select_ind))

    ids, hws, rayos, rayds, rgbs, alphas, pred_alphas, xyzs, normals = \
        tf.concat(ids, axis=0), tf.concat(hws, axis=0), tf.concat(rayos, axis=0), \
        tf.concat(rayds, axis=0), tf.concat(rgbs, axis=0), tf.concat(alphas, axis=0), \
        tf.concat(pred_alphas, axis=0), tf.concat(xyzs, axis=0), tf.concat(normals, axis=0)
    if data_type == 'nerf': lvises = tf.concat(lvises, axis=0)

    print('# Total VQ test samples:', tf.shape(ids)[0])

    if data_type == 'nerf':
        return ids, hws, rayos, rayds, rgbs, alphas, pred_alphas, xyzs, normals, lvises
    else:
        return ids, hws, rayos, rayds, rgbs, alphas, pred_alphas, xyzs, normals


def sample_tensor(tensor, mask, select_ind):
    tensor = tf.boolean_mask(tensor, mask)
    return tf.gather_nd(tensor, select_ind[:, None])


def train_iter(model, batch, optimizer, global_bs, thres=None):
    assert model.trainable_registered, \
        "Register the trainable layers before using `trainable_variables`"

    with tf.GradientTape(persistent=True) as tape:
        pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train', thres=thres)

        loss_kwargs['keep_batch'] = True  # keep the batch dimension
        per_example_loss, loss_dict = model.compute_loss(pred, gt, **loss_kwargs)
        weighted_loss = tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=global_bs)
    grads = tape.gradient(weighted_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return weighted_loss, partial_to_vis, loss_dict


def vali_iter(model, batch, global_bs, thres=None, full_vis=False):
    pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali', thres=thres, full_vis=full_vis)

    loss_kwargs['keep_batch'] = True  # keep the batch dimension
    per_example_loss, loss_dict = model.compute_loss(pred, gt, **loss_kwargs)
    weighted_loss = tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=global_bs)
    return weighted_loss, partial_to_vis, loss_dict


def vali_vq(model, batch, thres=None, full_vis=False):
    pred, gt, loss_kwargs, _ = model.vq_test(batch, mode='vali', thres=thres)

    loss_kwargs['keep_batch'] = True  # keep the batch dimension
    per_example_loss, loss_dict = model.compute_loss(pred, gt, **loss_kwargs)
    return loss_dict


if __name__ == '__main__':
    app.run(main)
