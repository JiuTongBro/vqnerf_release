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

# pylint: disable=invalid-unary-operand-type

from os.path import dirname, join, basename
import numpy as np
import tensorflow as tf
import cv2
import os

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil
from nerfactor.datasets.base import Dataset as BaseDataset
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil, \
    img as imgutil

logger = logutil.Logger(loggee="datasets/nerf_shape")



class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False):
        self.meta2buf = {}
        super().__init__(
            config, mode, debug=debug)
        seed = config.getint('DEFAULT', 'random_seed')
        tf.random.set_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    def _glob(self):
        root = self.config.get('DEFAULT', 'data_root')
        nerf_root = self.config.get('DEFAULT', 'data_nerf_root')
        self.data_type = self.config.get('DEFAULT', 'data_type')
        self.model_name = self.config.get('DEFAULT', 'model')

        metadata_paths, incomplete_paths = [], []
        # Shortcircuit if testing

        # Glob metadata paths
        if self.mode in ('train', 'render'): mode_str = 'train'
        else: mode_str = 'val'

        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(root, '%s_002' % mode_str)
        else:
            metadata_dir = join(root, '%s_???' % mode_str)

        # Include only cameras with all required buffers (depending on mode)
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            id_ = self._parse_id(metadata_path)
            normal_path = join(nerf_root, id_, 'normal.npy')
            xyz_path = join(nerf_root, id_, 'xyz.npy')
            alpha_path = join(nerf_root, id_, 'alpha.png')
            rgba_path = join(dirname(metadata_path), 'rgba.png')
            paths = {
                'xyz': xyz_path, 'normal': normal_path,
                'alpha': alpha_path, 'rgba': rgba_path}

            if self.data_type == 'nerf':
                lvis_path = join(nerf_root, id_, 'lvis.npy')
                paths['lvis'] = lvis_path

            if ioutil.all_exist(paths):
                metadata_paths.append(metadata_path)
                self.meta2buf[metadata_path] = paths
            else:
                incomplete_paths.append(metadata_path)
        if incomplete_paths:
            logger.warn((
                "Skipping\n\t%s\nbecause at least one of their paired "
                "buffers doesn't exist"), incomplete_paths)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    # pylint: disable=arguments-differ
    def _process_example_postcache(
            self, id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis=None):
        """Records image dimensions and samples rays.
        """
        hw = tf.shape(rgb)[:2]
        if self.data_type == 'nerf':
            if lvis is None: print('# NeRF Data Requires lvis!')
            rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis,\
                = self._sample_rays(rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis)
        else:
            rayo, rayd, rgb, alpha, pred_alpha, xyz, normal,\
                = self._sample_rays(rayo, rayd, rgb, alpha, pred_alpha, xyz, normal)
        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        if self.data_type == 'nerf':
            return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis
        else: return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal

    def _sample_rays(
            self, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis=None, alpha_thres=0.9):
        # outer sample for determination
        rayo = tf.reshape(rayo, (-1, 3))
        rayd = tf.reshape(rayd, (-1, 3))
        rgb = tf.reshape(rgb, (-1, 3))

        alpha = tf.reshape(alpha, (-1, 1))
        pred_alpha = tf.reshape(pred_alpha, (-1, 1))
        xyz = tf.reshape(xyz, (-1, 3))

        normal = tf.reshape(normal, (-1, 3))
        if self.data_type == 'nerf':
            if lvis is None: print('# NeRF Data Requires lvis!')
            lvis = tf.reshape(lvis, (-1, tf.shape(lvis)[2]))
            return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis
        else: return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal



    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        if self.data_type == 'nerf':
            id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis\
                = tf.py_function(self._load_data, [path], (
                    tf.string, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,))
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis
        else:
            id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal\
                = tf.py_function(self._load_data, [path], (
                    tf.string, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32,))
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal

    # pylint: disable=arguments-differ
    def _load_data(self, metadata_path):
        imh = self.config.getint('DEFAULT', 'imh')
        white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        use_nerf_alpha = self.config.getboolean('DEFAULT', 'use_nerf_alpha')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)

        # Rays
        metadata = ioutil.read_json(metadata_path)
        if self.data_type == 'dtu':

            self.k = imh / metadata['imh']
            imw = int(self.k * metadata['imw'])
            raw_world = np.array(metadata['world_mat'])
            raw_scale = np.array(metadata['scale_mat'])

            scaled_projection = (raw_world @ raw_scale)[0:3, 0:4]
            intrinsic, cam_to_world = self.decompose_projection_matrix(scaled_projection)
            intrinsic[:2, :3] = intrinsic[:2, :3] * self.k
            intrinsic_inv = np.linalg.inv(intrinsic)

            rayo, rayd = self._gen_rays(cam_to_world, intrinsic_inv, imh, imw)
            rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)

        else:
            imw = int(metadata['imw'] * imh / metadata['imh'])
            cam_to_world = np.array([
                float(x) for x in metadata['cam_transform_mat'].split(',')
            ]).reshape(4, 4)

            cam_angle_x = metadata['cam_angle_x']
            if 'cx' in metadata.keys():
                cx, cy = imh / metadata['imh'] * metadata['cx'], imh / metadata['imh'] * metadata['cy']
            else: cx, cy = None, None

            rayo, rayd = self._gen_rays(cam_to_world, cam_angle_x, imh, imw, cx, cy)
            rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)

        # Load precomputed shape properties from vanilla NeRF
        paths = self.meta2buf[metadata_path]
        xyz = ioutil.load_np(paths['xyz'])
        normal = ioutil.load_np(paths['normal'])

        # RGB and alpha, depending on the mode
        pred_alpha = xm.io.img.load(paths['alpha'])
        pred_alpha = xm.img.normalize_uint(pred_alpha)
        # Training or validation, where each camera has a paired image
        rgba = xm.io.img.load(paths['rgba'])
        assert rgba.ndim == 3 and rgba.shape[2] == 4, \
            "Input image is not RGBA"
        rgba = xm.img.normalize_uint(rgba)
        rgb = rgba[:, :, :3]
        # for test views, gt_alpha = pred_alpha
        if self.mode == 'test': alpha = pred_alpha
        else: alpha = rgba[:, :, 3]

        # Resize
        if imh != xyz.shape[0]:
            xyz = xm.img.resize(xyz, new_h=imh)
        if imh != normal.shape[0]:
            normal = xm.img.resize(normal, new_h=imh)
        if imh != alpha.shape[0]:
            alpha = xm.img.resize(alpha, new_h=imh)
        if imh != pred_alpha.shape[0]:
            pred_alpha = xm.img.resize(pred_alpha, new_h=imh)
        if imh != rgb.shape[0]:
            rgb = xm.img.resize(rgb, new_h=imh)

        '''
        # Resize
        if imh != xyz.shape[0]:
            imw = int(xyz.shape[1] * (imh / xyz.shape[0]))
            xyz = xm.img.resize(xyz, new_h=imh, new_w=imw)
        if imh != normal.shape[0]:
            imw = int(normal.shape[1] * (imh / normal.shape[0]))
            normal = xm.img.resize(normal, new_h=imh, new_w=imw)
        if imh != alpha.shape[0]:
            imw = int(alpha.shape[1] * (imh / alpha.shape[0]))
            alpha = xm.img.resize(alpha, new_h=imh, new_w=imw)
        if imh != pred_alpha.shape[0]:
            imw = int(pred_alpha.shape[1] * (imh / pred_alpha.shape[0]))
            pred_alpha = xm.img.resize(pred_alpha, new_h=imh, new_w=imw)
        if imh != rgb.shape[0]:
            imw = int(rgb.shape[1] * (imh / rgb.shape[0]))
            rgb = xm.img.resize(rgb, new_h=imh, new_w=imw)
        '''

        # Make sure there's no XYZ coinciding with camera (caused by occupancy
        # accumulating to 0)
        # assert not np.isclose(xyz, rayo).all(axis=2).any(), \
        #     "Found XYZs coinciding with the camera"

        # Deal collapsed point and cam
        zero_bg = tf.linalg.norm(xyz - rayo, axis=-1) == 0.
        xyz[zero_bg] = rayo[zero_bg] + rayd[zero_bg] * 0.1

        # Re-normalize normals and clip light visibility before returning
        zero_bg = np.mean(normal, axis=-1) == 0.
        normal[zero_bg] = np.array([0., 1., 0.])
        normal = xm.linalg.normalize(normal, axis=2)

        # Composite RGBA image onto white or black background
        bg = np.ones_like(rgb) if white_bg else np.zeros_like(rgb)
        rgb = imgutil.alpha_blend(rgb, alpha, tensor2=bg)
        rgb = rgb.astype(np.float32)

        if self.data_type == 'nerf':
            lvis = ioutil.load_np(paths['lvis'])
            if imh != lvis.shape[0]:
                lvis = xm.img.resize(lvis, new_h=imh)
            lvis = np.clip(lvis, 0, 1)
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis
        else:
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal

    # pylint: disable=arguments-differ
    def _gen_rays(self, to_world, intrinsic, imh, imw, cx=None, cy=None):
        # Ray origin
        cam_loc = to_world[:3, 3]
        rayo = np.tile( # (H * SPS, W * SPS, 3)
            cam_loc[None, None, :], (imh, imw, 1))
        # Ray directions
        xs = np.linspace(0, imw, imw, endpoint=False)
        ys = np.linspace(0, imh, imh, endpoint=False)
        xs, ys = np.meshgrid(xs, ys)
        # (0, 0)
        # +--------> (w, 0)
        # |           x
        # |
        # v y (0, h)
        if self.data_type == 'dtu':
            intrinsic_inv = intrinsic
            p = np.stack((xs, ys, np.ones_like(xs)), axis=-1) # local, [h,2,3]
            p = (intrinsic_inv[None, None, :3, :3] @ p[..., None])[..., 0] # h,w,3
            rayd = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)  # h, w, 3
            rayd = (to_world[None, None, :3, :3] @ rayd[..., None])[..., 0]
        else:
            angle_x = intrinsic
            fl = .5 * imw / np.tan(.5 * angle_x)
            if cx is None: cx = .5 * imw
            if cy is None: cy = .5 * imh
            rayd = np.stack(((xs - cx) / fl, -(ys - cy) / fl, -np.ones_like(xs)), axis=-1)  # local
            rayd = np.sum(rayd[:, :, np.newaxis, :] * to_world[:3, :3], axis=-1)  # world

        return rayo, rayd

    def decompose_projection_matrix(self, P):
        ''' Decompose intrincis and extrinsics from projection matrix (for Numpt object) '''
        # For Numpy object

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose

    @staticmethod
    def _parse_id(metadata_path): # pylint: disable=arguments-differ
        return basename(dirname(metadata_path))

    def get_n_views(self):
        if hasattr(self, 'files'):
            return len(self.files)
        raise RuntimeError("Call `_glob()` before `get_n_views()`")

    def _get_batch_size(self):
        if self.mode == 'train':
            bs = self.config.getint('DEFAULT', 'n_rays_per_step')
        else:
            # Total number of pixels is batch size, and will need to load
            # a datapoint to figure that out
            any_path = self.files[0]
            ret = self._load_data(any_path)
            map_data = ret[-1] # OK as long as shape is (H, W[, ?])
            bs = int(np.prod(map_data.shape[:2]))
        return bs
