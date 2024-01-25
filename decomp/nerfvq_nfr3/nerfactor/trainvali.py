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

from third_party.xiuminglib import xiuminglib as xm
from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil

tf_config=tf.compat.v1.ConfigProto()
# 设置最大占有GPU不超过显存的80%（可选）
tf_config.gpu_options.per_process_gpu_memory_fraction=0.8
tf_config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=tf_config)

flags.DEFINE_string(
    'config', 'gnerf.ini', "base .ini file in config/ or a full path")
flags.DEFINE_string('config_override', '', "e.g., 'key1=value1,key2=value2'")
flags.DEFINE_boolean('debug', False, "debug mode switch")
flags.DEFINE_enum(
    'device', 'gpu', ['cpu', 'gpu'], "running on what type of device(s)")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="trainvali")


def main(_):
    nfr_models = ['nerfactor', 'nfr_unit', 'ref_nfr']

    if FLAGS.debug:
        logger.warn("Debug mode: on")

    distributed_train_step_decor = distributed_train_step if FLAGS.debug \
        else tf.function(distributed_train_step)

    # Distribution strategy
    strategy = get_strategy()

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
    # ioutil.prepare_outdir(outdir, overwrite=False)
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
    datapipe_train = strategy.experimental_distribute_dataset(datapipe_train)

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
        vali_batches = strategy.experimental_distribute_dataset(vali_batches)

    with strategy.scope():
        # Model
        model_name = config.get('DEFAULT', 'model')
        Model = models.get_model_class(model_name)
        model = Model(config, debug=FLAGS.debug)
        model.register_trainable()
        if model_name in nfr_models:
            pretrain_epochs = int(config.get('DEFAULT', 'pretrain_epochs'))
            init_bias_weight = config.getfloat('DEFAULT', 'init_bias_weight')
            default_bias_weight = config.getfloat('DEFAULT', 'default_bias_weight')

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
        train_vis_epoch_dir = join(outdir, 'vis_train', 'epoch{e:09d}')
        vali_vis_epoch_dir = join(outdir, 'vis_vali', 'epoch{e:09d}')
        train_vis_epoch_dir_deque = deque([], keep_recent_epochs)
        vali_vis_epoch_dir_deque = deque([], keep_recent_epochs)
        train_vis_batch_rawf = join(
            train_vis_epoch_dir, 'batch{b:09d}_raw.pickle')
        vali_vis_batch_rawf = join(
            vali_vis_epoch_dir, 'batch{b:09d}_raw.pickle')
        train_vis_batch_dir = join(train_vis_epoch_dir, 'batch{b:09d}')
        vali_vis_batch_dir = join(vali_vis_epoch_dir, 'batch{b:09d}')

        # extension yourself in your overriding function (this makes the
        # pipeline general and not specific to any model)

        # ====== Training loop ======
        epochs = config.getint('DEFAULT', 'epochs')
        vis_train_batches = config.getint('DEFAULT', 'vis_train_batches')
        ckpt_period = config.getint('DEFAULT', 'ckpt_period')
        vali_period = config.getint('DEFAULT', 'vali_period')
        step_restored = ckpt.step.numpy()

        for _ in tqdm(range(step_restored, epochs), desc="Training epochs"):
            # ------ Train on all batches ------
            batch_loss, batch_vis, batch_time = [], [], []
            train_step = ckpt.step.numpy()

            if model_name in nfr_models:
                bias_weight, default_bias_weight = 1., 1.
                # if train_step == pretrain_epochs: model.pre2train()

            loss_dicts = []
            for batch_i, _batch in enumerate(datapipe_train):
                t0 = time()
                if dataset_name == 'shape_unit':
                    batch = outer_sample(_batch, config, data_type)
                else: batch = _batch

                if model_name in nfr_models:
                    if train_step < pretrain_epochs:
                        loss, to_vis, loss_dict = distributed_train_step_decor(
                            strategy, model, batch, optimizer, global_bs_train, True, bias_weight)
                    else:
                        loss, to_vis, loss_dict = distributed_train_step_decor(
                            strategy, model, batch, optimizer, global_bs_train, False, default_bias_weight)
                else:
                    loss, to_vis = distributed_train_step_decor(
                        strategy, model, batch, optimizer, global_bs_train)
                batch_time.append(time() - t0)
                batch_loss.append(loss)
                if model_name in nfr_models: loss_dicts.append(loss_dict)
                # if batch_i < vis_train_batches:
                #     batch_vis.append(to_vis)
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
                    vis_dirs = []
                    for batch_i, to_vis in enumerate(batch_vis):
                        raw_f = train_vis_batch_rawf.format(e=step, b=batch_i)
                        vis_dir = train_vis_batch_dir.format(e=step, b=batch_i)
                        model.vis_batch(
                            to_vis, vis_dir, mode='train', dump_raw_to=raw_f)
                        vis_dirs.append(vis_dir)

                maintain_epoch_queue(
                    train_vis_epoch_dir_deque,
                    train_vis_epoch_dir.format(e=step))

            # ------ Validation ------
            if vali_batches is not None and vali_period > 0 \
                    and step % vali_period == 0:

                if model_name == 'nerf' or model_name == 'gnerf':
                    ckpt_step = (step // ckpt_period) * ckpt_period
                else:
                    ckpt_step = step

                if model_name in nfr_models:
                    losses = loss_dicts[0]
                    for k,v in losses.items():
                        losses[k] = tf.reduce_mean(v).numpy()
                    for i in range(1, len(loss_dicts)):
                        for k,v in loss_dicts[i].items():
                            losses[k] += tf.reduce_mean(v).numpy()
                    for k,v in losses.items():
                        losses[k] = v.tolist()
                    loss_dir = vali_vis_epoch_dir.format(e=ckpt_step)
                    if not os.path.exists(loss_dir): os.makedirs(loss_dir)
                    with open(join(loss_dir, 'loss.json'), 'w') as f:
                        json.dump(losses, f)

                # Run validation on all validation batches
                batch_loss, batch_vis = [], []
                for batch_i, batch in enumerate(vali_batches):
                    # Validate on this validation batch
                    if model_name in nfr_models:
                        if train_step < pretrain_epochs:
                            loss, to_vis = distributed_vali_step(
                                strategy, model, batch, global_bs_vali, True, bias_weight)
                        else:
                            loss, to_vis = distributed_vali_step(
                                strategy, model, batch, global_bs_vali, False, default_bias_weight)
                    else:
                        loss, to_vis = distributed_vali_step(
                            strategy, model, batch, global_bs_vali)
                    batch_loss.append(loss)
                    batch_vis.append(to_vis)

                # Summarize/visualize validation
                with writer_vali.as_default():
                    tf.summary.scalar(
                        "loss_vali", tf.reduce_mean(batch_loss), step=ckpt_step)
                    vis_dirs = []
                    for batch_i, to_vis in enumerate(batch_vis):
                        raw_f = vali_vis_batch_rawf.format(e=ckpt_step, b=batch_i)
                        vis_dir = vali_vis_batch_dir.format(e=ckpt_step, b=batch_i)
                        model.vis_batch(
                            to_vis, vis_dir, mode='vali', dump_raw_to=raw_f)
                        vis_dirs.append(vis_dir)

                maintain_epoch_queue(
                    vali_vis_epoch_dir_deque, vali_vis_epoch_dir.format(e=ckpt_step))

        if model_name in ['nerfactor', 'gnerf', 'nerf']:
            save_metas(outdir)


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

    jitter_inds = tf.argmax(tf.reduce_max(tf.abs(rgb_jitters - rgb[None, 1:-1, 1:-1, :]), axis=-1), axis=0)
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


def save_metas(outdir):
    outdir = join(outdir, 'vis_vali')
    metrics = {'psnr':[],'ssim':[],'lpips':[],'psnr_luma':[],'ssim_luma':[],'mse':[]}
    for e_dir in os.listdir(outdir):
        epoch_metric = {'psnr':[],'ssim':[],'lpips':[],'psnr_luma':[],'ssim_luma':[],'mse':[]}
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


def get_strategy():
    """Creates a distributed strategy.
    """
    strategy = None
    if FLAGS.device == 'cpu':
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
    elif FLAGS.device == 'gpu':
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise NotImplementedError(FLAGS.device)
    return strategy


# May be decorated into a tf.function, depending on whether in debug mode
def distributed_train_step(strategy, model, batch, optimizer, global_bs, pretrain=False, bias_weight=None):
    assert model.trainable_registered, \
        "Register the trainable layers before using `trainable_variables`"

    def train_step(batch):
        with tf.GradientTape(persistent=True) as tape:
            if pretrain:
                if bias_weight is None:
                    pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train', pretrain=True)
                else: pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train', pretrain=True, bias_weight=bias_weight)
            else:
                if bias_weight is None:
                    pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train')
                else: pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='train', bias_weight=bias_weight)
            loss_kwargs['keep_batch'] = True  # keep the batch dimension
            if bias_weight is None: per_example_loss = model.compute_loss(pred, gt, **loss_kwargs)
            else: per_example_loss, loss_dict = model.compute_loss(pred, gt, **loss_kwargs)
            weighted_loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_bs)
        grads = tape.gradient(weighted_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if bias_weight is None: return weighted_loss, partial_to_vis
        else: return weighted_loss, partial_to_vis, loss_dict

    # Each GPU takes a step
    if bias_weight is None: 
        weighted_loss, partial_to_vis = strategy.run(train_step, args=(batch,))
        # Aggregate across GPUs
        loss, to_vis = aggeregate_dstributed(
            strategy, weighted_loss, partial_to_vis)
        return loss, to_vis
    else: 
        weighted_loss, partial_to_vis, loss_dict = strategy.run(train_step, args=(batch,))
        loss, to_vis, loss_dict = aggeregate_dstributed(
            strategy, weighted_loss, partial_to_vis, loss_dict)
        return loss, to_vis, loss_dict


# Not using tf.function for validation step because it can become very slow
# when there is a long loop. Given validation step is likely called relatively
# infrequently, eager should be fine
def distributed_vali_step(strategy, model, batch, global_bs, pretrain=False, bias_weight=None):
    def vali_step(batch):
        if pretrain:
            if bias_weight is None:
                pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali', pretrain=True)
            else: pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali', pretrain=True, bias_weight=bias_weight)
        else:
            if bias_weight is None:
                pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali')
            else: pred, gt, loss_kwargs, partial_to_vis = model(batch, mode='vali', bias_weight=bias_weight)
        loss_kwargs['keep_batch'] = True  # keep the batch dimension
        per_example_loss = model.compute_loss(pred, gt, **loss_kwargs)
        weighted_loss = tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=global_bs)
        return weighted_loss, partial_to_vis

    # Each GPU takes a step
    weighted_loss, partial_to_vis = strategy.run(vali_step, args=(batch,))

    # Aggregate across GPUs
    loss, to_vis = aggeregate_dstributed(
        strategy, weighted_loss, partial_to_vis)

    return loss, to_vis


def aggeregate_dstributed(strategy, weighted_loss, partial_to_vis, partial_dict=None):
    # Sum the weighted loss
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, weighted_loss, axis=None)

    # Concatenate the items to visualize back to the full batch
    to_vis = {}
    for k, v in partial_to_vis.items():
        to_vis[k] = tf.concat(
            tf.nest.flatten(v, expand_composites=True), axis=0)

    if partial_dict is not None:
        loss_dict = {}
        for k, v in partial_dict.items():
            loss_dict[k] = tf.concat(
                tf.nest.flatten(v, expand_composites=True), axis=0)

        return loss, to_vis, loss_dict
    else: return loss, to_vis


def maintain_epoch_queue(queue, new_epoch_dir):
    queue.appendleft(new_epoch_dir)
    for epoch_dir in xm.os.sortglob(dirname(new_epoch_dir), '*'):
        if epoch_dir not in queue:  # already evicted from queue (FIFO)
            rmtree(epoch_dir)


if __name__ == '__main__':
    app.run(main)
