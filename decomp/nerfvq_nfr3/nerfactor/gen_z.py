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

from nerfactor import datasets
from nerfactor import models
from nerfactor.util import logging as logutil, io as ioutil, \
    config as configutil
from third_party.xiuminglib import xiuminglib as xm
from third_party.turbo_colormap import turbo_colormap_data, interpolate_or_clip
import os


flags.DEFINE_string(
    'ckpt', '/path/to/ckpt-100', "path to checkpoint (prefix only)")
flags.DEFINE_string('function', None, "[render_edit, relight]")
flags.DEFINE_integer(
    'sv_axis_i', 0, "along which axis we do spatially-varying edits")
flags.DEFINE_float(
    'sv_axis_min', -1.5, "axis minimum for spatially-varying edits")
flags.DEFINE_float(
    'sv_axis_max', 1.5, "axis maximum for spatially-varying edits")
flags.DEFINE_string('tgt_albedo', None, "albedo edit name")
flags.DEFINE_string('tgt_brdf', None, "BRDF edit name")
flags.DEFINE_boolean('debug', False, "debug mode switch")
FLAGS = flags.FLAGS

logger = logutil.Logger(loggee="test")

def main(_):
    if FLAGS.debug:
        logger.warn("Debug mode: on")

    # Config
    config_ini = configutil.get_config_ini(FLAGS.ckpt)
    config = ioutil.read_config(config_ini)

    # Output directory
    outroot = join(config_ini[:-4], 'vis_test', basename(FLAGS.ckpt))

    # Make dataset
    logger.info("Making the actual data pipeline")
    dataset_name = config.get('DEFAULT', 'dataset')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, 'test', debug=FLAGS.debug)
    n_views = dataset.get_n_views()
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Restore model
    logger.info("Restoring trained model")
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config, debug=FLAGS.debug)
    ioutil.restore_model(model, FLAGS.ckpt)

    func = 'val_mat' # [gen_z, gen_mat, val_mat]

    # For all test views
    logger.info("Running inference")
    outroot = outroot.replace('vis_test', 'vis_z')

    for batch_i, batch in enumerate(
            tqdm(datapipe, desc="Inferring Views", total=n_views)):

        to_vis = model.gen_z(batch, genz=(func=='gen_z'))
        # Visualize
        outdir_scaled = join(outroot, 'val_{i:03d}'.format(i=batch_i))
        if not exists(outdir_scaled): makedirs(outdir_scaled)
        model.vis_batch(to_vis, outdir_scaled, mode='test')

    if func == 'gen_mat':
        dataset = Dataset(config, 'render', debug=FLAGS.debug)
        n_views = dataset.get_n_views()
        datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

        for batch_i, batch in enumerate(
                tqdm(datapipe, desc="Inferring Views", total=n_views)):

            to_vis = model.gen_z(batch)
            # Visualize
            outdir_scaled = join(outroot, 'train_{i:03d}'.format(i=batch_i))
            if not exists(outdir_scaled): makedirs(outdir_scaled)
            model.vis_batch(to_vis, outdir_scaled, mode='test')

if __name__ == '__main__':
    app.run(main)
