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

# pylint: disable=arguments-differ

from os.path import basename, dirname, join, exists
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.models.shape import Model as ShapeModel
from nerfactor.networks import mlp
from nerfactor.networks.embedder import Embedder
from nerfactor.util import vis as visutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil, img as imgutil, \
    math as mathutil, geom as geomutil, microfacet as micro_util


class Model(ShapeModel):
    def __init__(self, config, debug=False):

        self.data_type = config.get('DEFAULT', 'data_type')

        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        self.z_dim = config.getint('DEFAULT', 'conv_width')
        self.normalize_brdf_z = config.getboolean(
            'DEFAULT', 'normalize_z')

        # By now we have all attributes required by parent init.
        super().__init__(config, debug=debug)

        # Lighting, not readed, but tf.Variable
        self._light = None  # see the light property
        if not self.data_type == 'nerf':
            self._gamma_index, self._gamma_bias = None, None

        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        # independent from lighting source, just lighting directions
        lxyz, lareas = gen_light_xyz(*self.light_res)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        self.lareas = tf.convert_to_tensor(lareas, dtype=tf.float32)
        # Novel lighting conditions for relighting at test time:
        olat_inten = self.config.getfloat('DEFAULT', 'olat_inten', fallback=200)
        ambi_inten = self.config.getfloat(
            'DEFAULT', 'ambient_inten', fallback=0)
        # (1) OLAT
        novel_olat = OrderedDict()
        light_shape = self.light_res + (3,)
        if self.white_bg:
            # Add some ambient lighting to better match perception
            ambient = ambi_inten * tf.ones(light_shape, dtype=tf.float32)
        else:
            ambient = tf.zeros(light_shape, dtype=tf.float32)
        for i in [4]:
            for j in [0, 8, 16, 24]:
                one_hot = tutil.one_hot_img(*ambient.shape, i, j)
                envmap = olat_inten * one_hot + ambient
                novel_olat['%04d-%04d' % (i, j)] = envmap
        self.novel_olat = novel_olat

        # light_path = self.config.get('DEFAULT', 'light_path')
        # self.light = self._load_light(light_path)

        # (2) Light probes
        novel_probes = OrderedDict()
        test_envmap_dir = self.config.get('DEFAULT', 'test_envmap_dir')
        for path in xm.os.sortglob(test_envmap_dir, ext=('hdr', 'exr')):
            name = basename(path)[:-len('.hdr')]
            envmap = self._load_light(path)
            novel_probes[name] = envmap
        self.novel_probes = novel_probes
        # Tonemap and visualize these novel lighting conditions
        self.embed_light_h = self.config.getint(
            'DEFAULT', 'embed_light_h', fallback=32)
        self.novel_olat_uint = {}
        for k, v in self.novel_olat.items():
            vis_light = lightutil.vis_light(v, h=self.embed_light_h)
            self.novel_olat_uint[k] = vis_light
        self.novel_probes_uint = {}
        for k, v in self.novel_probes.items():
            vis_light = lightutil.vis_light(v, h=self.embed_light_h)
            self.novel_probes_uint[k] = vis_light
        # PSNR calculator
        self.psnr_luma = xm.metric.PSNR_luma('uint8')

    def _init_embedder(self):
        embedder = super()._init_embedder()
        return embedder

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        net = {}

        conv_width = 256
        net['diff_out'] = mlp.Network([conv_width, conv_width // 2, 3],
                                      act=['relu'] * 2 + ['sigmoid'], skip_at=[1])
        net['spec_out'] = mlp.Network([conv_width, conv_width // 2, 1],
                                      act=['relu'] * 2 + ['sigmoid'], skip_at=[1])
        net['rough_out'] = mlp.Network([conv_width, conv_width // 2, 1],
                                       act=['relu'] * 2 + ['sigmoid'], skip_at=[1])
        net['fine_enc'] = mlp.Network([mlp_width] * 4, act=['relu'] * 4, skip_at=[2])
        net['bottleneck'] = mlp.Network([mlp_width] + [conv_width] * 2, act=[None, 'relu', 'sigmoid'])

        print('\n#--- Registered Layers ---#')
        for sub_net in net.keys():
            for i, layer in enumerate(net[sub_net].layers):
                print(sub_net + '_layer%d' % i, ': ', layer.trainable)

        return net

    def _load_light(self, path):
        ext = basename(path).split('.')[-1]
        if ext == 'exr':
            arr = xm.io.exr.read(path)
        elif ext == 'hdr':
            arr = xm.io.hdr.read(path)
        elif ext == 'npy':
            arr = np.load(path)
        else:
            raise NotImplementedError(ext)
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        resized = imgutil.resize(tensor, new_h=self.light_res[0])
        return resized

    def gen_z(self, batch, genz=False):
        if self.data_type == 'nerf':
            id_, hw, _, _, _, alpha, pred_alpha, xyz, _, _ = batch
        else:
            id_, hw, _, _, _, alpha, pred_alpha, xyz, _ = batch
        # Mask out 100% background
        mask = alpha[:, 0] > 0
        xyz = tf.boolean_mask(xyz, mask)

        z_bias = self._pred_bias_at(xyz)

        basecolor = self._pred_diff_at(z_bias)
        # ------ BRDFs
        rough = self._pred_rough_at(z_bias)
        ks = self._pred_spec_at(z_bias)

        spec = ks * basecolor
        albedo = (1 - ks) * basecolor

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays
        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        spec = tf.scatter_nd(ind, spec, (n, 3))
        rough = tf.scatter_nd(ind, rough, (n, 1))

        if genz: z_bias = tf.scatter_nd(ind, z_bias, (n, self.z_dim))

        to_vis = {'id': id_, 'hw': hw,
                  'gt_alpha': alpha, 'pred_alpha': pred_alpha,
                  'albedo': albedo, 'spec': spec, 'rough': rough,}

        if genz: to_vis['z_bias'] = z_bias
        return to_vis

    def call(
            self, batch, mode='train', pretrain=False, relight_olat=False, relight_probes=False,
            save_z=False, opt_scale=None, bias_weight=None):
        self._validate_mode(mode)

        if self.data_type == 'nerf':
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
        else:
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
            lvis = None

        gt = {'rgb': rgb, 'normal': normal, 'alpha': alpha, 'xyz': xyz}

        # Mask out 100% background
        mask = alpha[:, 0] > 0
        rayo = tf.boolean_mask(rayo, mask)
        rgb = tf.boolean_mask(rgb, mask)
        xyz = tf.boolean_mask(xyz, mask)
        normal = tf.boolean_mask(normal, mask)

        if self.data_type == 'nerf':
            lvis = tf.boolean_mask(lvis, mask)

        # Directions
        surf2c = self._calc_vdir(rayo, xyz)
        surf2l = self._calc_ldir(xyz)

        normal_pred = self._normal_correct(normal, surf2c)

        z_bias = self._pred_bias_at(xyz)

        # ------ Albedo
        basecolor = self._pred_diff_at(z_bias)
        # ------ BRDFs
        rough = self._pred_rough_at(z_bias)
        ks = self._pred_spec_at(z_bias)
        spec = ks * basecolor
        albedo = (1 - ks) * basecolor

        brdf, brdf_spec, brdf_diff = self._eval_brdf_at(
            surf2l, surf2c, normal_pred, albedo, spec, rough)  # NxLx3

        # ------ Rendering equation
        rgb_pred, _, _ = self._render(  # all Nx3
            brdf, surf2l, normal_pred, lvis)

        if not mode == 'train':
            rgb_diff, _, _ = self._render(  # all Nx3
                brdf_diff, surf2l, normal_pred, lvis)

            rgb_spec, _, _ = self._render(  # all Nx3
                brdf_spec, surf2l, normal_pred, lvis)

        loss_kwargs = {
            'mode': mode, 'pretrain': pretrain,
            'gtc': rgb, 'rgb': rgb_pred, 'env': self._light,
            'spec': spec, 'rough': rough}

        # Put values back into the full shape
        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays
        if self.data_type == 'nerf':
            rgb_pred = imgutil.linear2srgb(rgb_pred)
        rgb_pred = tf.scatter_nd(ind, rgb_pred, (n, 3))

        xyz = tf.scatter_nd(ind, xyz, (n, 3))
        normal_pred = tf.scatter_nd(ind, normal_pred, (n, 3))
        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        spec = tf.scatter_nd(ind, spec, (n, 3))
        rough = tf.scatter_nd(ind, rough, (n, 1))
        ks = tf.scatter_nd(ind, ks, (n, 1))
        basecolor = tf.scatter_nd(ind, basecolor, (n, 3))

        if not mode == 'train':
            rgb_diff = tf.scatter_nd(ind, rgb_diff, (n, 3))
            rgb_spec = tf.scatter_nd(ind, rgb_spec, (n, 3))

        # ------ Loss
        pred = {
            'rgb': rgb_pred, 'normal': normal_pred, 'albedo': albedo, 'basecolor': basecolor,
            'alpha': pred_alpha, 'spec': spec, 'rough': rough, 'ks': ks, 'xyz': xyz}

        if not mode == 'train':
            pred['rgb_spec'] = rgb_spec
            pred['rgb_diff'] = rgb_diff

        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def _render(
            self, brdf, l, n, light_vis=None,
            white_light_override=False):
        # l:light, n:normal
        linear2srgb = self.config.getboolean('DEFAULT', 'linear2srgb')
        light = self.light
        if white_light_override:
            light = np.ones_like(self.light)
        cos = tf.einsum('ijk,ik->ij', l, n)  # NxL
        # Areas for intergration
        areas = tf.reshape(self.lareas, (1, -1, 1))  # 1xLx1
        # NOTE: unnecessary if light_vis already encodes it, but won't hurt
        front_lit = tf.cast(cos > 0, tf.float32)

        if light_vis is None: lvis = front_lit
        else: lvis = front_lit * light_vis # NxL

        def integrate(light):
            light_flat = tf.reshape(light, (-1, 3))  # Lx3
            light = lvis[:, :, None] * light_flat[None, :, :]  # NxLx3
            light_pix_contrib = brdf * light * cos[:, :, None] * areas  # NxLx3
            rgb = tf.reduce_sum(light_pix_contrib, axis=1)  # Nx3, sum of integration
            if not self.data_type == 'nerf':
                rgb = (rgb * self.gamma[0]) ** self.gamma[1]
            # Tonemapping
            rgb = tfp.math.clip_by_value_preserve_gradient(rgb, 0., 1.)  # NOTE
            # Colorspace transform

            return rgb

        # ------ Render under original lighting
        rgb = integrate(light)

        return rgb, None, None  # Nx3

    # Following NeILF, use the learnable scaling for real data to mitigate the color-space gap between render and display 
    @property
    def gamma(self):
        if self._gamma_index is None:  # initialize just once
            gamma_index = tf.convert_to_tensor(np.array([1.]).astype(np.float32))
            self._gamma_index = tf.Variable(gamma_index, trainable=True, dtype=tf.float32)
            gamma_bias = tf.convert_to_tensor(np.array([1.]).astype(np.float32))
            self._gamma_bias = tf.Variable(gamma_bias, trainable=True, dtype=tf.float32)
        _index = tfp.math.clip_by_value_preserve_gradient(self._gamma_index, 0., 5.)  # clip to prevent too large scale
        # No negative light
        return tf.concat([self._gamma_bias, _index], axis=0)

    @property
    def light(self):
        if self._light is None:  # initialize just once
            inten = self.config.getfloat('DEFAULT', 'light_init_val')
            light = tf.ones(self.light_res + (3,)) * inten
            self._light = tf.Variable(light, trainable=True)
        # No negative light
        return tfp.math.clip_by_value_preserve_gradient(self._light, 0., np.inf)  # 3D

    def _pred_bias_at(self, pts):
        xyz_mlp = self.net['fine_enc']
        xyz_out = self.net['bottleneck']
        embedder = self.embedder['xyz']

        def chunk_func(surf):
            surf_embed = embedder(surf)
            y = xyz_mlp(surf_embed)
            z_bias = xyz_out(y)  # norm later
            return z_bias

        z_bias = self.chunk_apply(chunk_func, pts, self.z_dim, self.mlp_chunk)
        z_bias = tf.debugging.check_numerics(z_bias, "Z")
        return z_bias  # Nxz_dim

    def _pred_diff_at(self, z):  # finetune
        # Given that albedo generally ranges from 0.1 to 0.8
        albedo_scale = self.config.getfloat(
            'DEFAULT', 'albedo_slope', fallback=1.)
        albedo_bias = self.config.getfloat(
            'DEFAULT', 'albedo_bias', fallback=0.)

        diff_out = self.net['diff_out']  # output in [0, 1]

        def chunk_func(z_):
            albedo = diff_out(z_)
            return albedo

        albedo = self.chunk_apply(chunk_func, z, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias  # [bias, scale + bias]
        albedo = tf.debugging.check_numerics(albedo, "Albedo")
        return albedo  # Nx3

    def _pred_spec_at(self, z):  # finetune
        spec_out = self.net['spec_out']  # output in [0, 1]

        def chunk_func(z_):
            spec = spec_out(z_)
            return spec

        spec = self.chunk_apply(chunk_func, z, 1, self.mlp_chunk)
        spec = tf.debugging.check_numerics(spec, "Specular")
        return spec  # Nx3

    def _pred_rough_at(self, z):  # finetune
        rough_out = self.net['rough_out']  # output in [0, 1]

        def chunk_func(z_):
            rough = rough_out(z_)
            return rough

        rough = self.chunk_apply(chunk_func, z, 1, self.mlp_chunk)
        rough = tf.debugging.check_numerics(rough, "Roughness")
        return rough  # Nx1

    def _normal_correct(self, normal, surf2c):  # finetune
        cos = tf.reduce_sum(normal * surf2c, axis=-1, keepdims=True)
        normal_c = tf.where(cos>=0, normal, -normal)
        return normal_c  # Nx1

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, spec, rough):
        brdf, brdf_spec, brdf_diff = micro_util.get_brdf(pts2l, pts2c, normal, albedo=albedo, rough=rough, f0=spec)
        return brdf, brdf_spec, brdf_diff  # NxLx3

    def compute_loss(self, pred, gt, **kwargs):
        """Additional priors on light probes.
        """
        mode = kwargs.pop('mode')
        lambert_weight = self.config.getfloat('DEFAULT', 'lambert_weight')

        #
        rgb_gt = kwargs.pop('gtc')
        rgb_pred = kwargs.pop('rgb')

        if self.data_type == 'nerf':
            linear_gt = imgutil.srgb2linear(rgb_gt)
            srgb_pred = imgutil.linear2srgb(rgb_pred)
        else:
            linear_gt = rgb_gt
            srgb_pred = rgb_pred

        loss_dict = {}
        # RGB recon. loss is always here
        loss_dict['rgb'] = tf.keras.losses.MSE(linear_gt, rgb_pred)
        loss = loss_dict['rgb']  # N
        # If validation, just MSE -- return immediately
        if not mode == 'train': return loss
        '''
        if lambert_weight > 0:
            spec, rough = kwargs.pop('spec'), kwargs.pop('rough')

            # lambert_restrict = tf.reduce_mean(spec, axis=-1) + 1. - tf.reduce_mean(rough, axis=-1)
            sg_rough = tf.stop_gradient(rough)
            sg_rough = tf.where(sg_rough<0.5, 0., 2*sg_rough-1.)
            lambert_restrict = tf.reduce_max(spec, axis=-1) * sg_rough
            loss_dict['lambert'] = lambert_weight * lambert_restrict
            loss += loss_dict['lambert']
        '''
        loss_dict['loss'] = loss
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss, loss_dict

    def _brdf_prop_as_img(self, brdf_prop):
        """Z in learned BRDF.

        Input and output are both NumPy arrays, not tensors.
        """
        # Get min. and max. from seen BRDF Zs
        seen_z = self.brdf_model.latent_code.z
        seen_z = seen_z.numpy()
        seen_z_rgb = seen_z[:, :3]
        min_ = seen_z_rgb.min()
        max_ = seen_z_rgb.max()
        range_ = max_ - min_
        assert range_ > 0, "Range of seen BRDF Zs is 0"
        # Clip predicted values and scale them to [0, 1]
        z_rgb = brdf_prop[:, :, :3]
        z_rgb = np.clip(z_rgb, min_, max_)
        z_rgb = (z_rgb - min_) / range_
        return z_rgb

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            light_vis_h=256, olat_vis=False, alpha_thres=0.8):
        # Visualize estimated lighting
        if mode == 'vali':
            # The same for all batches/views, so do it just once
            np_light_path = join(dirname(outdir), 'np_light.npy')
            light_vis_path = join(dirname(outdir), 'pred_light.png')
            if not exists(light_vis_path):
                light = self.light
                np.save(np_light_path, light.numpy())
                lightutil.vis_light(
                    light, outpath=light_vis_path, h=light_vis_h)

        # Do what parent does
        self._validate_mode(mode)
        # Shortcircuit if training because rays are randomly sampled and
        # therefore very likely don't form a complete image
        if mode == 'train':
            return
        hw = data_dict.pop('hw')[0, :]
        hw = tuple(hw.numpy())
        id_ = data_dict.pop('id')[0]
        id_ = tutil.eager_tensor_to_str(id_)
        # To NumPy and reshape back to images
        # data_dict:{'pred_{}','gt_{}'}
        for k, v in data_dict.items():
            if v is None:
                continue  # no-op
            v_ = v.numpy()
            if k in ('pred_rgb_olat', 'pred_rgb_probes'):
                v_ = v_.reshape(hw + (v_.shape[1], 3))
            elif k.endswith(('rgb', 'albedo', 'normal', 'rgb_diff', 'rgb_spec', 'basecolor',
                             'brdf', 'brdf_spec', 'brdf_diff', 'spec', 'xyz', 'stop')):
                v_ = v_.reshape(hw + (3,))
            elif k.endswith(('occu', 'depth', 'disp', 'alpha', 'rough', 'conv', 'ks')):
                v_ = v_.reshape(hw)
            elif k.endswith('lvis'):
                v_ = v_.reshape(hw + (v_.shape[1],))
            elif k.endswith(('ref', 'z_bias')):
                v_ = v_.reshape(hw + (self.z_dim,))
            else:
                raise NotImplementedError(k)
            data_dict[k] = v_

        # Write images
        img_dict = {}

        alpha = data_dict['gt_alpha']
        alpha[alpha < alpha_thres] = 0  # stricter compositing
        pd_alpha = data_dict['pred_alpha']
        pd_alpha[pd_alpha < alpha_thres] = 0  # stricter compositing

        for k, v in data_dict.items():
            # OLAT-relit RGB
            if k == 'pred_rgb_olat':  # HxWxLx3
                if v is None:
                    continue
                olat_names = list(self.novel_olat.keys())
                olat_first_n = np.prod(self.light_res) // 2  # top half only
                olat_n = 0
                for i, lname in enumerate(
                        tqdm(olat_names, desc="Writing OLAT-Relit Results")):
                    # Skip visiualization if enough OLATs
                    if olat_n >= olat_first_n:
                        continue
                    else:
                        olat_n += 1
                    #
                    k_relit = k + '_' + lname
                    v_relit = v[:, :, i, :]
                    light_uint = self.novel_olat_uint[lname]
                    # img = composite_on_avg_light(v_relit, light_uint)
                    bg = np.ones_like(v_relit) if self.white_bg else np.zeros_like(v_relit)
                    img = imgutil.alpha_blend(v_relit, alpha, bg)
                    img_dict[k_relit] = xm.io.img.write_arr(
                        img, join(outdir, k_relit + '.png'), clip=True)
            # Light probe-relit RGB
            elif k == 'pred_rgb_probes':  # HxWxLx3
                if v is None:
                    continue
                probe_names = list(self.novel_probes.keys())
                for i, lname in enumerate(probe_names):
                    k_relit = k + '_' + lname
                    v_relit = v[:, :, i, :]
                    light_uint = self.novel_probes_uint[lname]
                    # img = composite_on_avg_light(v_relit, light_uint)
                    bg = np.ones_like(v_relit) if self.white_bg else np.zeros_like(v_relit)
                    img = imgutil.alpha_blend(v_relit, alpha, bg)
                    img_dict[k_relit] = xm.io.img.write_arr(
                        img, join(outdir, k_relit + '.png'), clip=True)
            # RGB
            elif k.endswith(('rgb',)):  # HxWx3
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('basecolor', 'albedo', 'normal', 'ks', 'rough', 'spec')):  # HxWx3
                if not exists(outdir): os.makedirs(outdir)
                np.save(join(outdir, k + '.npy'), v)
                img_dict[k] = xm.io.img.write_arr(
                    v, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('stop',)):  # HxWx3
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('conv', 'ks',)):  # HxWx3
                img_dict[k] = xm.io.img.write_arr(
                    v, join(outdir, k + '.png'), clip=True)

            elif not (mode == 'render'):
                # Normals
                if k.endswith(('xyz')):
                    v_ = (v + 1) / 2  # [-1, 1] to [0, 1]
                    # v_ = v
                    bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                    img = imgutil.alpha_blend(v_, alpha, bg)
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
                # Light visibility
                elif k.endswith('lvis'):
                    mean = np.mean(v, axis=2)  # NOTE: average across all lights
                    bg = np.ones_like(mean) if self.white_bg \
                        else np.zeros_like(mean)
                    img = imgutil.alpha_blend(mean, alpha, bg)
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
                    # Optionally, visualize per-light vis.
                    if olat_vis:
                        for i in tqdm(
                                range(4 if self.debug else v.shape[2] // 2),  # half
                                desc="Writing Per-Light Visibility (%s)" % k):
                            v_olat = v[:, :, i]
                            ij = np.unravel_index(i, self.light_res)
                            k_olat = k + '_olat_%04d-%04d' % ij
                            img = imgutil.alpha_blend(v_olat, alpha, bg)
                            img_dict[k_olat] = xm.io.img.write_arr(
                                img, join(outdir, k_olat + '.png'), clip=True)
                elif k.endswith(('rgb_diff', 'rgb_spec', 'brdf', 'brdf_spec', 'brdf_diff')):  # HxWx3
                    if not exists(outdir): os.makedirs(outdir)
                    np.save(join(outdir, k + '.npy'), v)
                    bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                    img = imgutil.alpha_blend(v, alpha, bg)
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
                elif k.endswith(('ref', 'z_bias')):  # HxWx3
                    if not exists(outdir): os.makedirs(outdir)
                    np.save(join(outdir, k + '.npy'), v)
                # Everything else
                else:
                    img = v
                    img_dict[k] = xm.io.img.write_arr(
                        img, join(outdir, k + '.png'), clip=True)
        # Shortcircuit if testing because there will be no ground truth for
        # us to make .apng comparisons
        if mode in ('test', 'render'):
            # Write metadata that doesn't require ground truth (e.g., view name)
            metadata = {'id': id_}
            ioutil.write_json(metadata, join(outdir, 'metadata.json'))
            return

        # Write metadata (e.g., view name, PSNR, etc.)
        # official realization of NeRFactor
        psnr_luma = self.psnr_luma(img_dict['gt_rgb'], img_dict['pred_rgb']).tolist()

        metadata = {'id': id_, 'psnr_luma': psnr_luma}
        ioutil.write_json(metadata, join(outdir, 'metadata.json'))

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train', fps=12):
        viewer_prefix = self.config.get('DEFAULT', 'viewer_prefix')
        self._validate_mode(mode)
        # Shortcircuit if training (same reason as above)
        if mode == 'train':
            return None
        # Validation or testing
        else:
            outpath = outpref + '.html'
            self._compile_into_webpage(batch_vis_dirs, outpath)
        '''
        else:
            outpath = outpref + '.mp4'
            self._compile_into_video(batch_vis_dirs, outpath, fps=fps)
        '''
        view_at = viewer_prefix + outpath
        return view_at  # to be logged into TensorBoard

    def _compile_into_webpage(self, batch_dirs, out_html):
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            metadata = str(metadata)
            row = [
                metadata,
                join(batch_dir, 'pred-vs-gt_rgb.apng'),
                join(batch_dir, 'pred_rgb.png'),
                join(batch_dir, 'pred_albedo.png'),
                join(batch_dir, 'pred_brdf.png')]
            rowcaps = [
                "Metadata", "RGB", "RGB (pred.)", "Albedo (pred.)",
                "BRDF (pred.)"]
            rowtypes = ['text', 'image', 'image', 'image', 'image']
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_normal.png'))
                rowcaps.append("Normal (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_normal.apng'))
                rowcaps.append("Normal")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_normal.png'))
                rowcaps.append("Normal (pred.)")
                rowtypes.append('image')
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_lvis.png'))
                rowcaps.append("Light Visibility (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_lvis.apng'))
                rowcaps.append("Light Visibility")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_lvis.png'))
                rowcaps.append("Light Visibility (pred.)")
                rowtypes.append('image')
            #
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Write HTML
        bg_color = 'white' if self.white_bg else 'black'
        text_color = 'black' if self.white_bg else 'white'
        html = xm.vis.html.HTML(bgcolor=bg_color, text_color=text_color)
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)

    def _compile_into_video(self, batch_dirs, out_mp4, fps=12):
        data_root = self.config.get('DEFAULT', 'data_root')
        # Assume batch directory order is the right view order
        batch_dirs = sorted(batch_dirs)
        if self.debug:
            batch_dirs = batch_dirs[:10]
        # Tonemap and visualize all lighting conditions used
        orig_light_uint = lightutil.vis_light(self.light, h=self.embed_light_h)
        frames = []
        # View synthesis
        for batch_dir in tqdm(batch_dirs, desc="View Synthesis"):
            frame = visutil.make_frame(
                batch_dir,
                (('normal', 'lvis', 'nn'), ('brdf', 'albedo', 'rgb')),
                data_root=data_root, put_text_param=self.put_text_param,
                rgb_embed_light=orig_light_uint)
            # To guard against missing buffer, which makes the frame None
            if frame is not None:
                frames.append(frame)
        # Relighting
        relight_view_dir = batch_dirs[-1]  # fixed to the final view
        lvis_paths = xm.os.sortglob(relight_view_dir, 'pred_lvis_olat*.png')
        for lvis_path in tqdm(lvis_paths, desc="Final View, OLAT"):
            olat_id = basename(lvis_path)[len('pred_lvis_olat_'):-len('.png')]
            if self.debug and (olat_id not in self.novel_probes_uint):
                continue
            frame = visutil.make_frame(
                relight_view_dir,
                (('normal', f'lvis_olat_{olat_id}', 'nn'),
                 ('brdf', 'albedo', f'rgb_olat_{olat_id}')),
                data_root=data_root, put_text_param=self.put_text_param,
                rgb_embed_light=self.novel_olat_uint[olat_id])
            # To guard against missing buffer, which makes the frame None
            if frame is not None:
                frames.append(frame)
        # Simultaneous
        envmap_names = list(self.novel_probes.keys())
        n_envmaps = len(envmap_names)
        batch_dirs_roundtrip = list(reversed(batch_dirs)) + batch_dirs
        batch_dirs_roundtrip += batch_dirs_roundtrip  # 2nd roundtrip
        n_views_per_envmap = len(batch_dirs_roundtrip) / n_envmaps  # float
        map_i = 0
        for view_i, batch_dir in enumerate(
                tqdm(batch_dirs_roundtrip, desc="View Roundtrip, IBL")):
            envmap_name = envmap_names[map_i]
            frame = visutil.make_frame(
                batch_dir,
                (('normal', 'lvis', 'nn'),
                 ('brdf', 'albedo', f'rgb_probes_{envmap_name}')),
                data_root=data_root, put_text_param=self.put_text_param,
                rgb_embed_light=self.novel_probes_uint[envmap_name])
            # To guard against missing buffer, which makes the frame None
            if frame is not None:
                frames.append(frame)
            # Time to switch to the next map?
            if (view_i + 1) > n_views_per_envmap * (map_i + 1):
                map_i += 1
        #
        xm.vis.video.make_video(frames, outpath=out_mp4, fps=fps)
