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

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil
from nerfactor.datasets.base import Dataset as BaseDataset
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil, \
    img as imgutil

logger = logutil.Logger(loggee="datasets/nerf_shape")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, interp=1):
        self.meta2buf = {}
        self.interp = interp
        super().__init__(
            config, mode, debug=debug)

    def _glob(self):
        nerf_root = self.config.get('DEFAULT', 'data_nerf_root')
        nerf_root = nerf_root.replace('surf', 'video')
        self.data_type = self.config.get('DEFAULT', 'data_type')

        metadata_paths, incomplete_paths = [], []
        # Shortcircuit if testing

        # Glob metadata paths
        mode_str = 'test'

        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(nerf_root, '%s_002' % mode_str)
        else:
            metadata_dir = join(nerf_root, '%s_???' % mode_str)

        print(metadata_dir)

        # Include only cameras with all required buffers (depending on mode)
        metadatas = xm.os.sortglob(metadata_dir, 'metadata.json')
        for i in range(0, len(metadatas), self.interp):
            metadata_path = metadatas[i]
            id_ = self._parse_id(metadata_path)
            normal_path = join(nerf_root, id_, 'normal.npy')
            xyz_path = join(nerf_root, id_, 'xyz.npy')
            alpha_path = join(nerf_root, id_, 'alpha.png')
            basecolor_path = join(nerf_root, id_, 'rgb.png')
            paths = {
                'xyz': xyz_path, 'normal': normal_path,
                'alpha': alpha_path, 'basecolor': basecolor_path}

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
            self, id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis=None):
        """Records image dimensions and samples rays.
        """
        hw = tf.shape(rgb)[:2]
        if self.data_type == 'nerf':
            if lvis is None: print('# NeRF Data Requires lvis!')
            rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis,\
                = self._sample_rays(rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis)
        else:
            rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref,\
                = self._sample_rays(rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref)
        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        if self.data_type == 'nerf':
            return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis
        else: return id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref

    def _sample_rays(
            self, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis=None, alpha_thres=0.9):
        # Shortcircuit if need all rays
        if self.mode in ('vali', 'test', 'render'):
            rayo = tf.reshape(rayo, (-1, 3))
            rayd = tf.reshape(rayd, (-1, 3))
            rgb = tf.reshape(rgb, (-1, 3))

            alpha = tf.reshape(alpha, (-1, 1))
            pred_alpha = tf.reshape(pred_alpha, (-1, 1))
            xyz = tf.reshape(xyz, (-1, 3))

            normal = tf.reshape(normal, (-1, 3))
            ref = tf.reshape(ref, (-1, 3))

            if self.data_type == 'nerf':
                if lvis is None: print('# NeRF Data Requires lvis!')
                lvis = tf.reshape(lvis, (-1, tf.shape(lvis)[2]))
                return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis
            else: return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref

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
        select_ind = tf.random.uniform(
            (self.bs,), minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
        select_coords = tf.gather_nd(coords, select_ind[:, None])  # n,2
        select_jitter = tf.gather_nd(coords_jitter, select_ind[:, None])  # n,2
        select_ind = tf.concat([select_coords, select_jitter], axis=-1)
        select_ind = tf.reshape(select_ind, (-1, 2))  # [p1, p1_n, p2, p2_n, ...]

        rayo = tf.gather_nd(rayo, select_ind)
        rayd = tf.gather_nd(rayd, select_ind)
        rgb = tf.gather_nd(rgb, select_ind)
        alpha = tf.gather_nd(alpha, select_ind)
        alpha = tf.reshape(alpha, (-1, 1))
        pred_alpha = tf.gather_nd(pred_alpha, select_ind)
        pred_alpha = tf.reshape(pred_alpha, (-1, 1))
        xyz = tf.gather_nd(xyz, select_ind)
        normal = tf.gather_nd(normal, select_ind)
        ref = tf.gather_nd(ref, select_ind)

        if self.data_type == 'nerf':
            if lvis is None: print('# NeRF Data Requires lvis!')
            lvis = tf.gather_nd(lvis, select_ind)
            return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis
        else: return rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        if self.data_type == 'nerf':
            id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis\
                = tf.py_function(self._load_data, [path], (
                    tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,))
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref, lvis
        else:
            id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref\
                = tf.py_function(self._load_data, [path], (
                    tf.string, tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,))
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, ref

    # pylint: disable=arguments-differ
    def _load_data(self, metadata_path):
        white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)

        # Rays
        metadata = ioutil.read_json(metadata_path)
        # Load precomputed shape properties from vanilla NeRF
        paths = self.meta2buf[metadata_path]
        xyz = ioutil.load_np(paths['xyz'])
        normal = ioutil.load_np(paths['normal'])

        imh, imw = xyz.shape[:2]

        if self.data_type == 'dtu':

            # Rays
            intrinsic = np.array(metadata['intrinsic']).reshape(4, 4)
            cam_to_world = np.array(metadata['cam_transform_mat']).reshape(4, 4)
            intrinsic_inv = np.linalg.inv(intrinsic)

            rayo, rayd = self._gen_rays(cam_to_world, intrinsic_inv, imh, imw)
            rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)

        else:

            cam_to_world = np.array(metadata['cam_transform_mat']).reshape(4, 4)

            focal = metadata['focal']
            if 'cx' in metadata.keys():
                cx, cy = metadata['cx'], metadata['cy']
            else:
                cx, cy = None, None

            rayo, rayd = self._gen_rays(cam_to_world, focal, imh, imw, cx, cy)
            rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)

        # RGB and alpha, depending on the mode
        pred_alpha = xm.io.img.load(paths['alpha'])
        pred_alpha = xm.img.normalize_uint(pred_alpha)
        alpha = pred_alpha

        basecolor = xm.io.img.load(paths['basecolor'])
        basecolor = xm.img.normalize_uint(basecolor)
        rgb = basecolor

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
        if imh != basecolor.shape[0]:
            basecolor = xm.img.resize(basecolor, new_h=imh)

        # Make sure there's no XYZ coinciding with camera (caused by occupancy
        # accumulating to 0)
        # assert not np.isclose(xyz, rayo).all(axis=2).any(), \
        #     "Found XYZs coinciding with the camera"
        # Re-normalize normals and clip light visibility before returning
        zero_bg = np.mean(normal, axis=-1) == 0.
        normal[zero_bg] = np.array([0., 1., 0.])
        far_bg = ~np.isclose(np.linalg.norm(normal, axis=2), 1)
        normal[far_bg] = np.array([0., 1., 0.])
        normal = xm.linalg.normalize(normal, axis=2)
        assert np.isclose(np.linalg.norm(normal, axis=2), 1).all(), \
            "Found normals with a norm far away from 1"

        # Composite RGBA image onto white or black background
        bg = np.ones_like(rgb) if white_bg else np.zeros_like(rgb)
        rgb = imgutil.alpha_blend(rgb, alpha, tensor2=bg)
        rgb = rgb.astype(np.float32)

        if self.data_type == 'nerf':
            lvis = ioutil.load_np(paths['lvis'])
            if imh != lvis.shape[0]:
                lvis = xm.img.resize(lvis, new_h=imh)
            lvis = np.clip(lvis, 0, 1)
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, basecolor, lvis
        else:
            return id_, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, basecolor

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
            fl = intrinsic
            if cx is None: cx = .5 * imw
            if cy is None: cy = .5 * imh
            rayd = np.stack(((xs - cx) / fl, -(ys - cy) / fl, -np.ones_like(xs)), axis=-1)  # local
            rayd = np.sum(rayd[:, :, np.newaxis, :] * to_world[:3, :3], axis=-1)  # world

        return rayo, rayd

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

