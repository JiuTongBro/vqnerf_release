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
import cv2

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.models.nfr_unit import Model as NfrModel
from nerfactor.models.shape import Model as ShapeModel
from nerfactor.networks import mlp
from nerfactor.networks.vq_layers import VectorQuantizerEMA
from nerfactor.util import vis as visutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil, img as imgutil, \
    math as mathutil, geom as geomutil, microfacet as micro_util


class Model(ShapeModel):
    def __init__(self, config, debug=False):

        self.data_type = config.get('DEFAULT', 'data_type')
        self.no_brdf_chunk = config.getboolean('DEFAULT', 'no_brdf_chunk', fallback=True)
        print('# No BRDF Chunk? ', self.no_brdf_chunk)

        self.seed = config.getint('DEFAULT', 'random_seed')

        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        self.z_dim = config.getint('DEFAULT', 'conv_width')

        self.nfr_ckpt = config.get('DEFAULT', 'nfr_model_ckpt')
        nfr_config_path = configutil.get_config_ini(self.nfr_ckpt)
        self.config_nfr = ioutil.read_config(nfr_config_path)

        # By now we have all attributes required by parent init.
        super().__init__(config, debug=debug)

        # BRDF
        self.l_var_weight = config.getfloat(
            'DEFAULT', 'l_var_weight')
        self.brdf_chunk_size = self.config.getint(
            'DEFAULT', 'brdf_chunk_size', fallback=50000)

        # Lighting, not readed, but tf.Variable
        self._light, self._codebook = None, None  # see the light property
        if not self.data_type == 'nerf':
            self._gamma_index, self._gamma_bias = None, None

        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        # independent from lighting source, just lighting directions
        lxyz, lareas = gen_light_xyz(*self.light_res)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        self.lareas = tf.convert_to_tensor(lareas, dtype=tf.float32)
        '''
        nfr_out_dir = dirname(dirname(self.nfr_ckpt))
        vali_dir = join(nfr_out_dir, 'vis_vali')
        epoch_dirs = xm.os.sortglob(vali_dir, 'epoch?????????')
        epoch_dir = epoch_dirs[-1]
        light_path = join(epoch_dir, 'np_light.npy')
        print('Load Light From: ', light_path)
        self.light = self._load_light(light_path)
        '''
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
        print('Test Envs: ', test_envmap_dir)

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
        self.psnr = xm.metric.PSNR('uint8')

    def _init_embedder(self):
        embedder = super()._init_embedder()
        return embedder

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        self.num_embed = self.config.getint('DEFAULT', 'num_embed')
        commitment_cost = self.config.getfloat('DEFAULT', 'commitment_cost')
        net = {}

        net['diff_vq'] = mlp.Network([self.z_dim, self.z_dim // 2, 3],
                                      act=['relu'] * 2 + ['sigmoid'], skip_at=[1])
        net['spec_vq'] = mlp.Network([self.z_dim, self.z_dim // 2, 3],
                                      act=['relu'] * 2 + ['sigmoid'], skip_at=[1])
        net['rough_vq'] = mlp.Network([self.z_dim, self.z_dim // 2, 1],
                                       act=['relu'] * 2 + ['sigmoid'], skip_at=[1])

        nfr_main = NfrModel(self.config_nfr)
        ioutil.restore_model(nfr_main, self.nfr_ckpt)
        nfr_main.trainable = True
        net['fine_enc'] = nfr_main.net['fine_enc']
        net['bottleneck'] = nfr_main.net['bottleneck']
        net['diff_main'] = nfr_main.net['diff_out']
        net['spec_main'] = nfr_main.net['spec_out']
        net['rough_main'] = nfr_main.net['rough_out']

        print('\n#--- Registered Layers ---#')
        for sub_net in net.keys():
            for i, layer in enumerate(net[sub_net].layers):
                print(sub_net + '_layer%d' % i, ': ', layer.trainable)

        self.vq_layer = VectorQuantizerEMA(embedding_dim=self.z_dim, num_embeddings=self.num_embed,
                                        commitment_cost=commitment_cost, seed=self.seed)
        return net

    def _load_light(self, path, resize=False):
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
        if resize: resized = imgutil.resize(tensor, new_h=self.light_res[0])
        else: resized = tensor

        return resized
    '''
    def adjust_light(self, light, base_inten=1.):
        # base_inten=1 is useful
        light_h, light_w = light.shape[:2]
        light_mean, light_std = np.mean(light), np.std(light)
        upper_light = cv2.resize(light[:light_h // 2, ...], (light_w, 3 * light_h // 4))
        lower_light = cv2.resize(light[light_h // 2:, ...], (light_w, light_h // 4))
        new_light = np.concatenate([upper_light, lower_light], axis=0)
        new_light = np.clip(new_light, 0., light_mean + 3 * light_std) + base_inten
        new_light = new_light + base_inten
        print('Adjust light...')
        light_mean = np.mean(light)
        new_light = light + light_mean * 0.5
        return new_light
    '''

    def init_z(self, batch):
        if self.data_type == 'nerf':
            id_, hw, _, _, _, alpha, pred_alpha, xyz, _, _ = batch
        else:
            id_, hw, _, _, _, alpha, pred_alpha, xyz, _ = batch
        # Mask out 100% background
        mask = alpha[:, 0] > 0
        xyz = tf.boolean_mask(xyz, mask)

        z_pred = self._pred_enc_at(xyz)

        to_vis = {'id': id_, 'hw': hw, 'z_pred': z_pred}
        return to_vis

    def init_mat(self, z_pred):

        rough = self._pred_rough_at(z_pred)
        basecolor = self._pred_diff_at(z_pred)
        ks = self._pred_spec_at(z_pred)

        spec = ks * basecolor
        albedo = (1 - ks) * basecolor
        mat = tf.concat([albedo, spec, rough], axis=-1)

        return mat

    def fast_embed(
            self, batch, mode='train', thres=None, ref_batch=True):

        self._validate_mode(mode)

        if ref_batch:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _ = batch
        else:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
        # Mask out 100% background
        mask = alpha[:, 0] > 0
        xyz = tf.boolean_mask(xyz, mask)

        z_enc = self._pred_enc_at(xyz)

        if thres is not None:
            thres = tf.convert_to_tensor(thres, dtype=tf.float32)
            thres = tf.reshape(thres, (1, self.num_embed))

        z_norm = mathutil.safe_l2_normalize(z_enc, axis=1)
        codebook = self.get_codebook()
        vq_outs = self.vq_layer(z_norm, codebook, is_training=(mode=='train'), thres=thres)
        z_vq, vq_loss, embed_ind = vq_outs['quantize'], vq_outs['loss'], vq_outs['encoding_indices']+1

        loss_kwargs = {'mode': mode}
        # ------ Loss
        pred = {'alpha': pred_alpha}
        gt = {'alpha': alpha}

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays

        # ------ To visualize
        embed = tf.scatter_nd(ind, embed_ind[:, None], (n, 1))
        xyz = tf.scatter_nd(ind, xyz, (n, 3))
        to_vis = {'id': id_, 'hw': hw, 'embed': embed, 'xyz': xyz,}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v

        return pred, gt, loss_kwargs, to_vis

    def _update_material(self, src, mask, update):
        update = tf.convert_to_tensor([update], dtype=tf.float32)
        return src * (1. - mask) + mask * update

    def fast_render(
            self, batch, mode='train',
            relight_olat=False, relight_probes=False, opt_scale=None,
            edit_mask=None, edit_material=None, ref_batch=False, dst_env=None,
            gen_embed=False, thres=None, ):

        self._validate_mode(mode)

        if not ref_batch:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
                lvis = None
        else:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _, = batch
                lvis = None

        # Mask out 100% background
        mask = alpha[:, 0] > 0
        rayo = tf.boolean_mask(rayo, mask)
        rgb = tf.boolean_mask(rgb, mask)
        xyz = tf.boolean_mask(xyz, mask)
        normal = tf.boolean_mask(normal, mask)

        if self.data_type == 'nerf':
            lvis = tf.boolean_mask(lvis, mask)

        if edit_mask is not None:
            edit_mask = tf.boolean_mask(edit_mask, mask)[..., 0:1] > 0.
            edit_mask = tf.cast(edit_mask, dtype=tf.float32)

        # Directions
        surf2l = self._calc_ldir(xyz)
        surf2c = self._calc_vdir(rayo, xyz)

        normal_pred, normal_jitter = self._normal_correct(normal, surf2c), None

        z_enc = self._pred_enc_at(xyz)

        if gen_embed:

            if thres is not None:
                thres = tf.convert_to_tensor(thres, dtype=tf.float32)
                thres = tf.reshape(thres, (1, self.num_embed))

            z_norm = mathutil.safe_l2_normalize(z_enc, axis=1)
            codebook = self.get_codebook()
            vq_outs = self.vq_layer(z_norm, codebook, is_training=(mode == 'train'), thres=thres)
            z_vq, vq_loss, embed_ind = vq_outs['quantize'], vq_outs['loss'], vq_outs['encoding_indices'] + 1

        # sg_enc = tf.stop_gradient(z_enc)
        rough = self._pred_rough_at(z_enc)
        basecolor = self._pred_diff_at(z_enc)
        ks = self._pred_spec_at(z_enc)

        spec = ks * basecolor
        albedo = (1 - ks) * basecolor

        if edit_mask is not None:
            if not edit_material['diff'][0] < 0:
                albedo = self._update_material(albedo, edit_mask, edit_material['diff'])
            if not edit_material['spec'][0] < 0:
                spec = self._update_material(spec, edit_mask, edit_material['spec'])
            if not edit_material['rough'][0] < 0:
                rough = self._update_material(rough, edit_mask, edit_material['rough'])

        if opt_scale is not None:
            scaled_albedo = albedo * opt_scale
            scaled_spec = spec * opt_scale
        else: scaled_albedo, scaled_spec = albedo, spec

        brdf, _, _ = self._eval_brdf_at(
            surf2l, surf2c, normal_pred, scaled_albedo, scaled_spec, rough)  # NxLx3

        rgb_pred, rgb_olat, rgb_probes = self._render(  # all Nx3
            brdf, surf2l, normal_pred, lvis,
            relight_olat=relight_olat, relight_probes=relight_probes,
            dst_env=dst_env)

        loss_kwargs = {'mode': mode, 'gtc': rgb,}

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays
        if rgb_olat is not None:
            if self.data_type == 'nerf':
                rgb_olat = imgutil.linear2srgb(rgb_olat)
            rgb_olat = tf.scatter_nd(
                ind, rgb_olat, (n, len(self.novel_olat), 3))
        if rgb_probes is not None:
            if self.data_type == 'nerf':
                rgb_probes = imgutil.linear2srgb(rgb_probes)
            rgb_probes = tf.scatter_nd(
                ind, rgb_probes, (n, len(self.novel_probes), 3))
        rgb = tf.scatter_nd(ind, rgb, (n, 3))

        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        spec = tf.scatter_nd(ind, spec, (n, 3))
        rough = tf.scatter_nd(ind, rough, (n, 1))

        # ------ Loss
        pred = {'alpha': pred_alpha, 'albedo': albedo, 'spec': spec, 'rough': rough}

        if gen_embed:
            embed = tf.scatter_nd(ind, embed_ind[:, None], (n, 1))
            pred['embed'] = embed

        if dst_env is not None:
            if self.data_type == 'nerf':
                rgb_pred = imgutil.linear2srgb(rgb_pred)
            rgb_pred = tf.scatter_nd(ind, rgb_pred, (n, 3))
            pred['rgb'] = rgb_pred

        if rgb_olat is not None:
            pred['rgb_olat'] = rgb_olat
        if rgb_probes is not None:
            pred['rgb_probes'] = rgb_probes

        gt = {'rgb': rgb,  'alpha': alpha,}

        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def vis_mat(
            self, batch, mode='train', opt_scale=None, ref_batch=False, thres=None):

        self._validate_mode(mode)

        if not ref_batch:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
                lvis = None
        else:
            if self.data_type == 'nerf':
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _, lvis = batch
            else:
                id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, _, = batch
                lvis = None

        # Mask out 100% background
        mask = alpha[:, 0] > 0
        xyz = tf.boolean_mask(xyz, mask)

        z_enc = self._pred_enc_at(xyz)

        if thres is not None:
            thres = tf.convert_to_tensor(thres, dtype=tf.float32)
            thres = tf.reshape(thres, (1, self.num_embed))

        z_norm = mathutil.safe_l2_normalize(z_enc, axis=1)
        codebook = self.get_codebook()
        vq_outs = self.vq_layer(z_norm, codebook, is_training=(mode=='train'), thres=thres)
        z_vq, vq_loss, embed_ind = vq_outs['quantize'], vq_outs['loss'], vq_outs['encoding_indices']+1

        # sg_enc = tf.stop_gradient(z_enc)
        rough = self._pred_rough_at(z_enc)
        basecolor = self._pred_diff_at(z_enc)
        ks = self._pred_spec_at(z_enc)

        spec = ks * basecolor
        albedo = (1 - ks) * basecolor

        loss_kwargs = {'mode': mode,}

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays

        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        spec = tf.scatter_nd(ind, spec, (n, 3))
        rough = tf.scatter_nd(ind, rough, (n, 1))

        # ------ Loss
        pred = {'alpha': pred_alpha, 'albedo': albedo,
                'spec': spec, 'rough': rough}

        embed = tf.scatter_nd(ind, embed_ind[:, None], (n, 1))
        pred['embed'] = embed

        gt = { 'alpha': alpha,}

        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def vq_test(
            self, batch, mode='vali', thres=None,):

        if self.data_type == 'nerf':
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
        else:
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
            lvis = None

        # Mask out 100% background
        mask = alpha[:, 0] > 0
        rayo = tf.boolean_mask(rayo, mask)
        rgb = tf.boolean_mask(rgb, mask)
        xyz = tf.boolean_mask(xyz, mask)
        normal = tf.boolean_mask(normal, mask)

        if self.data_type == 'nerf':
            lvis = tf.boolean_mask(lvis, mask)

        # Directions
        surf2l = self._calc_ldir(xyz)
        surf2c = self._calc_vdir(rayo, xyz)

        normal_pred, normal_jitter = self._normal_correct(normal, surf2c), None

        z_enc = self._pred_enc_at(xyz)

        if thres is not None:
            thres = tf.convert_to_tensor(thres, dtype=tf.float32)
            thres = tf.reshape(thres, (1, self.num_embed))

        z_norm = mathutil.safe_l2_normalize(z_enc, axis=1)
        codebook = self.get_codebook()
        vq_outs = self.vq_layer(z_norm, codebook, is_training=(mode=='train'), thres=thres)
        z_vq, vq_loss, embed_ind = vq_outs['quantize'], vq_outs['loss'], vq_outs['encoding_indices']+1
        usage = tf.where(tf.reduce_max(vq_outs['encodings'], axis=0, keepdims=True) > 0, 1., 0.) # 1, n_vq

        vq_rough = self._pred_rough_at(z_vq, vq=True)
        vq_albedo = self._pred_diff_at(z_vq, vq=True)
        vq_spec = self._pred_spec_at(z_vq, vq=True)

        if mode == 'train' or self.no_brdf_chunk:
            vq_brdf, _, _ = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, vq_albedo, vq_spec, vq_rough)  # NxLx3
        else:
            vq_brdf, _, _ = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, vq_albedo, vq_spec, vq_rough,
                chunk_size=self.brdf_chunk_size)  # NxLx3

        # ------ Rendering equation
        vq_rgb, _, _ = self._render(  # all Nx3
            vq_brdf, surf2l, normal_pred, lvis,
            relight_olat=False, relight_probes=False)

        loss_kwargs = {
            'vqloss': vq_loss, 'vqrgb': vq_rgb,
            'mode': mode, 'gtc': rgb, 'rgb': vq_rgb,
            'usage': usage}

        # ------ Loss
        pred = {'alpha': pred_alpha,}
        gt = {'alpha': alpha,}
        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}

        return pred, gt, loss_kwargs, to_vis

    def call(
            self, batch, mode='train', thres=None, full_vis=False):

        self._validate_mode(mode)
        '''
        print('\n# ---- Weight Visualization:')
        print(self.net['fine_enc'].layers[0].weights[0][0][0])
        print(self.net['diff_vq'].layers[0].weights)
        '''
        if self.data_type == 'nerf':
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal, lvis = batch
        else:
            id_, hw, rayo, rayd, rgb, alpha, pred_alpha, xyz, normal = batch
            lvis = None

        # Mask out 100% background
        mask = alpha[:, 0] > 0
        rayo = tf.boolean_mask(rayo, mask)
        rgb = tf.boolean_mask(rgb, mask)
        xyz = tf.boolean_mask(xyz, mask)
        normal = tf.boolean_mask(normal, mask)

        # print('# ----  First XYZ Value: ', xyz[0][0])

        if self.data_type == 'nerf':
            lvis = tf.boolean_mask(lvis, mask)

        # Directions
        surf2l = self._calc_ldir(xyz)
        surf2c = self._calc_vdir(rayo, xyz)

        normal_pred, normal_jitter = self._normal_correct(normal, surf2c), None

        z_enc = self._pred_enc_at(xyz)

        if thres is not None:
            thres = tf.convert_to_tensor(thres, dtype=tf.float32)
            thres = tf.reshape(thres, (1, self.num_embed))

        # if not mode == 'train':
        #     print('# ----  First VQ Code Ind: ', self.codebook[0][0])
        z_norm = mathutil.safe_l2_normalize(z_enc, axis=1)
        codebook = self.get_codebook()
        vq_outs = self.vq_layer(z_norm, codebook, is_training=(mode=='train'), thres=thres)
        z_vq, vq_loss, embed_ind = vq_outs['quantize'], vq_outs['loss'], vq_outs['encoding_indices']+1

        # EMA Update
        # The VQ codebook is updated by EMA but not gradient propagation. So ignore relevant warnings.
        if mode == 'train':
            self._codebook.assign(vq_outs['update'])

        # sg_enc = tf.stop_gradient(z_enc)
        rough = self._pred_rough_at(z_enc)
        basecolor = self._pred_diff_at(z_enc)
        ks = self._pred_spec_at(z_enc)

        spec = ks * basecolor
        albedo = (1 - ks) * basecolor

        if mode == 'train' or self.no_brdf_chunk:
            brdf, brdf_spec, brdf_diff = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, albedo, spec, rough)  # NxLx3
        else:
            brdf, brdf_spec, brdf_diff = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, albedo, spec, rough,
                chunk_size=self.brdf_chunk_size)  # NxLx3

        # ------ Rendering equation
        rgb_pred, rgb_olat, rgb_probes = self._render(  # all Nx3
            brdf, surf2l, normal_pred, lvis)

        if not mode == 'train':
            rgb_diff, _, _ = self._render(  # all Nx3
                brdf_diff, surf2l, normal_pred, lvis)

            rgb_spec, _, _ = self._render(  # all Nx3
                brdf_spec, surf2l, normal_pred, lvis)

        vq_rough = self._pred_rough_at(z_vq, vq=True)
        vq_albedo = self._pred_diff_at(z_vq, vq=True)
        vq_spec = self._pred_spec_at(z_vq, vq=True)

        if mode == 'train' or self.no_brdf_chunk:
            vq_brdf, _, _ = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, vq_albedo, vq_spec, vq_rough)  # NxLx3
        else:
            vq_brdf, _, _ = self._eval_brdf_at(
                surf2l, surf2c, normal_pred, vq_albedo, vq_spec, vq_rough,
                chunk_size=self.brdf_chunk_size)  # NxLx3

        # ------ Rendering equation
        vq_rgb, _, _ = self._render(  # all Nx3
            vq_brdf, surf2l, normal_pred, lvis,
            relight_olat=False, relight_probes=False)

        loss_kwargs = {
            'vqloss': vq_loss, 'vqrgb': vq_rgb,
            'mode': mode, 'gtc': rgb, 'rgb': rgb_pred,
            'spec': spec, 'rough': rough,
            'z': z_vq, 'embed': self._codebook,}

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays
        # Put values back into the full shape
        if self.data_type == 'nerf':
            rgb_pred = imgutil.linear2srgb(rgb_pred)
        rgb_pred = tf.scatter_nd(ind, rgb_pred, (n, 3))

        normal_pred = tf.scatter_nd(ind, normal_pred, (n, 3))
        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        spec = tf.scatter_nd(ind, spec, (n, 3))
        rough = tf.scatter_nd(ind, rough, (n, 1))
        ks = tf.scatter_nd(ind, ks, (n, 1))

        if not mode == 'train':
            rgb_diff = tf.scatter_nd(ind, rgb_diff, (n, 3))
            rgb_spec = tf.scatter_nd(ind, rgb_spec, (n, 3))

        if full_vis: z_enc = tf.scatter_nd(ind, z_enc, (n, self.z_dim))

        rgb = tf.scatter_nd(ind, rgb, (n, 3))
        normal = tf.scatter_nd(ind, normal, (n, 3))

        # ------ Loss
        pred = {
            'rgb': rgb_pred, 'normal': normal_pred, 'albedo': albedo,
            'alpha': pred_alpha, 'spec': spec, 'rough': rough, 'ks': ks,}

        if not mode == 'train':
            pred['rgb_diff'] = rgb_diff
            pred['rgb_spec'] = rgb_spec

        gt = {'rgb': rgb, 'normal': normal, 'alpha': alpha}

        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        if full_vis: to_vis['enc_z'] = z_enc

        if not mode == 'train':
            embed = tf.scatter_nd(ind, embed_ind[:, None], (n, 1))
            pred['embed'] = embed

            if self.data_type == 'nerf':
                vq_rgb = imgutil.linear2srgb(vq_rgb)
            vq_rgb = tf.scatter_nd(ind, vq_rgb, (n, 3))
            vq_albedo = tf.scatter_nd(ind, vq_albedo, (n, 3))
            vq_spec = tf.scatter_nd(ind, vq_spec, (n, 3))
            vq_rough = tf.scatter_nd(ind, vq_rough, (n, 1))

            pred['vq_rgb'] = vq_rgb
            pred['vq_albedo'] = vq_albedo
            pred['vq_spec'] = vq_spec
            pred['vq_rough'] = vq_rough

        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def _render(
            self, brdf, l, n, light_vis=None,
            relight_olat=False, relight_probes=False,
            dst_env=None):
        # l:light, n:normal
        if dst_env is None: light = self.light
        else: light = self.novel_probes[dst_env]
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
            return rgb

        # ------ Render under original lighting
        rgb = integrate(light)

        rgb_probes = None
        if relight_probes:
            rgb_probes = []
            for _, light in self.novel_probes.items():
                rgb_relit = integrate(light)
                rgb_probes.append(rgb_relit)
            rgb_probes = tf.concat([x[:, None, :] for x in rgb_probes], axis=1)
            rgb_probes = tf.debugging.check_numerics(
                rgb_probes, "Light Probe Renders")
        return rgb, None, rgb_probes  # Nx3

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
            nfr_out_dir = dirname(dirname(self.nfr_ckpt))
            vali_dir = join(nfr_out_dir, 'vis_vali')
            epoch_dirs = xm.os.sortglob(vali_dir, 'epoch?????????')
            epoch_dir = epoch_dirs[-1]
            light_path = join(epoch_dir, 'np_light.npy')
            print('Load Light From: ', light_path)
            light = self._load_light(light_path)
            self._light = tf.Variable(light, trainable=True)
        # No negative light
        return tfp.math.clip_by_value_preserve_gradient(self._light, 0., np.inf)  # 3D

    def get_codebook(self):
        if self._codebook is None:  # initialize just once
            cluster_center_path = self.config.get('DEFAULT', 'cluster_center_path')
            cluster_center = np.load(cluster_center_path)
            codebook = tf.transpose(tf.convert_to_tensor(cluster_center, dtype=tf.float32))
            self._codebook = tf.Variable(codebook, trainable=True)
        code_dict = tfp.math.clip_by_value_preserve_gradient(self._codebook, 0., 1.)
        code_dict = mathutil.safe_l2_normalize(code_dict, axis=0)
        return code_dict # 3D

    def _pred_enc_at(self, pts):
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

    def _pred_diff_at(self, z, vq=False):  # finetune
        # Given that albedo generally ranges from 0.1 to 0.8
        albedo_scale = self.config.getfloat(
            'DEFAULT', 'albedo_slope', fallback=1.)
        albedo_bias = self.config.getfloat(
            'DEFAULT', 'albedo_bias', fallback=0.)

        if vq: diff_out = self.net['diff_vq']  # output in [0, 1]
        else: diff_out = self.net['diff_main']  # output in [0, 1]

        def chunk_func(z_):
            albedo = diff_out(z_)
            return albedo

        albedo = self.chunk_apply(chunk_func, z, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias  # [bias, scale + bias]
        albedo = tf.debugging.check_numerics(albedo, "Albedo")
        return albedo  # Nx3

    def _pred_spec_at(self, z, vq=False):  # finetune
        if vq: spec_out = self.net['spec_vq']  # output in [0, 1]
        else: spec_out = self.net['spec_main']  # output in [0, 1]

        def chunk_func(z_):
            spec = spec_out(z_)
            return spec

        if vq: spec = self.chunk_apply(chunk_func, z, 3, self.mlp_chunk)
        else: spec = self.chunk_apply(chunk_func, z, 1, self.mlp_chunk)
        spec = tf.debugging.check_numerics(spec, "Specular")
        return spec  # Nx3

    def _pred_rough_at(self, z, vq=False):  # finetune
        if vq: rough_out = self.net['rough_vq']  # output in [0, 1]
        else: rough_out = self.net['rough_main']  # output in [0, 1]

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

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, spec, rough, chunk_size=None):

        if chunk_size is None:
            brdf, brdf_spec, brdf_diff = micro_util.get_brdf(pts2l, pts2c, normal, albedo=albedo, rough=rough, f0=spec)
            return brdf, brdf_spec, brdf_diff  # NxLx3

        n, l = tf.shape(pts2l)[:2]
        brdf = tf.zeros((n * l, 3), dtype=tf.float32)
        brdf_spec = tf.zeros((n * l, 3), dtype=tf.float32)
        brdf_diff = tf.zeros((n * l, 3), dtype=tf.float32)

        for i in tf.range(0, n, chunk_size):
            end_i = tf.math.minimum(n, i + chunk_size)

            pts2l_chunk = pts2l[i:end_i]
            pts2c_chunk = pts2c[i:end_i]
            normal_chunk = normal[i:end_i]
            albedo_chunk = albedo[i:end_i]
            rough_chunk = rough[i:end_i]
            spec_chunk = spec[i:end_i]

            brdf_chunk, brdf_spec_chunk, brdf_diff_chunk = micro_util.get_brdf(
                pts2l_chunk, pts2c_chunk, normal_chunk, albedo=albedo_chunk, rough=rough_chunk, f0=spec_chunk)

            brdf_chunk = tf.reshape(brdf_chunk, (-1, 3))
            brdf_spec_chunk = tf.reshape(brdf_spec_chunk, (-1, 3))
            brdf_diff_chunk = tf.reshape(brdf_diff_chunk, (-1, 3))

            brdf = tf.tensor_scatter_nd_update(
                brdf, tf.range(i*l, end_i*l)[:,  None], brdf_chunk)
            brdf_spec = tf.tensor_scatter_nd_update(
                brdf_spec, tf.range(i*l, end_i*l)[:, None], brdf_spec_chunk)
            brdf_diff = tf.tensor_scatter_nd_update(
                brdf_diff, tf.range(i*l, end_i*l)[:, None], brdf_diff_chunk)

        brdf = tf.reshape(brdf, (n, l, 3))
        brdf_spec = tf.reshape(brdf_spec, (n, l, 3))
        brdf_diff = tf.reshape(brdf_diff, (n, l, 3))

        return brdf, brdf_spec, brdf_diff  # NxLx3

    def compute_loss(self, pred, gt, **kwargs):
        """Additional priors on light probes.
        """

        chr_alpha = self.config.getfloat('DEFAULT', 'chr_alpha')
        chr_thres = self.config.getfloat('DEFAULT', 'chr_thres')

        vq_loss_weight = self.config.getfloat('DEFAULT', 'vq_loss_weight')
        chromaticity_weight = self.config.getfloat('DEFAULT', 'chromaticity_loss_weight')
        mat_sloss_weight = self.config.getfloat('DEFAULT', 'mat_sloss_weight')
        combine_weight = self.config.getfloat('DEFAULT', 'combine_weight')
        sim_loss_weight = self.config.getfloat('DEFAULT', 'sim_loss_weight')
        lambert_weight = self.config.getfloat('DEFAULT', 'lambert_weight')

        #
        mode = kwargs.pop('mode')

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

        # If validation, just MSE -- return immediately
        if not mode == 'train':
            loss_dict['rgb'] = tf.keras.losses.MSE(rgb_gt, srgb_pred)
            loss = loss_dict['rgb']  # N

            vq_rgb = kwargs.pop('vqrgb')
            vq_srgb = imgutil.linear2srgb(vq_rgb)
            loss_dict['vqrgb'] = tf.keras.losses.MSE(rgb_gt, vq_srgb)
            loss += loss_dict['vqrgb']

            # chr
            chr_pd, lchr_gt = self._rgb2chromaticity(vq_rgb), self._rgb2chromaticity(linear_gt)
            loss_dict['chromaticity'] = tf.keras.losses.MSE(lchr_gt, chr_pd)
            loss += loss_dict['chromaticity']

            return loss, loss_dict


        # RGB recon. loss is always here
        loss_dict['rgb'] = combine_weight * tf.keras.losses.MSE(linear_gt, rgb_pred)
        loss = loss_dict['rgb']  # N

        vq_rgb = kwargs.pop('vqrgb')
        loss_dict['vqrgb'] = tf.keras.losses.MSE(linear_gt, vq_rgb)
        loss += loss_dict['vqrgb']

        vq_loss = kwargs.pop('vqloss')
        loss_dict['vqloss'] = vq_loss_weight * vq_loss
        loss += loss_dict['vqloss']

        schr_gt = self._rgb2chromaticity(rgb_gt)
        if chromaticity_weight > 0:
            chr_pd, lchr_gt = self._rgb2chromaticity(vq_rgb), self._rgb2chromaticity(linear_gt)
            loss_dict['chromaticity'] = chromaticity_weight * tf.keras.losses.MSE(lchr_gt, chr_pd)
            loss += loss_dict['chromaticity']

        # Smooth Loss
        if mat_sloss_weight > 0:
            z_vq = kwargs.pop('z')

            chr1, chr2 = schr_gt[::2, :], schr_gt[1::2, :]
            chr_e = tf.sqrt(tf.reduce_sum(tf.square(chr1 - chr2), axis=-1))
            chr_e = tf.where(chr_e > chr_thres, chr_e, 0.)
            w_chr = tf.exp(-chr_alpha * chr_e)

            mat1, mat2 = z_vq[::2, :], z_vq[1::2, :]
            chr_sl = w_chr * (1. - tf.reduce_sum(mat1 * mat2, axis=-1))
            # chr_sl = 1. - tf.reduce_sum(mat1 * mat2, axis=-1)

            chr_sl = tf.reshape(tf.concat([chr_sl[:, None], chr_sl[:, None]], axis=-1), (-1,))
            loss_dict['chr_smooth'] = mat_sloss_weight * chr_sl
            loss += loss_dict['chr_smooth']

        # A minor loss that prevents the VQ centers to be too close. 
        if sim_loss_weight > 0:
            codebook = tf.transpose(self.get_codebook())
            c1 = tf.repeat(codebook[:, None, :], self.num_embed, axis=1)
            c2 = tf.repeat(codebook[None, :, :], self.num_embed, axis=0)  # n,n,d
            dist = tf.sqrt(tf.reduce_sum(tf.square(c1 - c2), axis=-1))  # n,n

            max_value = tf.reduce_max(dist)
            diag_mask = tf.eye(self.num_embed)
            masked_dist = dist * (1 - diag_mask) + diag_mask * max_value
            sim = tf.reduce_min(masked_dist)
            sim_loss = - tf.math.log(sim)

            loss_dict['sim_smooth'] = sim_loss_weight * sim_loss
            loss += loss_dict['sim_smooth']

        if lambert_weight > 0:
            spec, rough = kwargs.pop('spec'), kwargs.pop('rough')

            # lambert_restrict = tf.reduce_mean(spec, axis=-1) + 1. - tf.reduce_mean(rough, axis=-1)
            sg_rough = tf.stop_gradient(rough)
            sg_rough = tf.where(sg_rough<0.5, 0., 2*sg_rough-1.)
            lambert_restrict = tf.reduce_max(spec, axis=-1) * sg_rough[:, 0]
            loss_dict['lambert'] = lambert_weight * lambert_restrict
            loss += loss_dict['lambert']

        loss_dict['loss'] = loss
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss, loss_dict

    def vis_batch(
            self, data_dict, outdir, mode='train', dump_raw_to=None,
            light_vis_h=256, alpha_thres=0.8, simp=False, full_vis_path=None):

        full_vis = full_vis_path is not None

        # Visualize estimated lighting
        if mode == 'vali':
            # The same for all batches/views, so do it just once
            np_light_path = join(dirname(outdir), 'np_light.npy')
            light_vis_path = join(dirname(outdir), 'pred_light.png')
            if not exists(light_vis_path):
                light = self.light
                lightutil.vis_light(
                    light, outpath=light_vis_path, h=light_vis_h)
                np.save(np_light_path, light.numpy())

            if full_vis:
                codebook = self.get_codebook()
                np.save(join(full_vis_path, 'vq_embed.npy'), codebook.numpy())

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
            elif k.endswith(('rgb', 'albedo', 'normal', 'diff', 'spec', 'xyz', 'basecolor')):
                v_ = v_.reshape(hw + (3,))
            elif k.endswith(('occu', 'depth', 'disp', 'alpha', 'rough', 'embed', 'ks')):
                v_ = v_.reshape(hw)
            elif k.endswith(('z',)):
                v_ = v_.reshape(hw + (v_.shape[1],))
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
            elif k.endswith(('albedo', 'spec',  'rough', 'ks', 'basecolor')):  # HxWx3
                if not exists(outdir): os.makedirs(outdir)
                np.save(join(outdir, k + '.npy'), v)
                img_dict[k] = xm.io.img.write_arr(
                    v, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('diff',)):  # HxWx3
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif k.endswith(('embed',)):  # HxWx3
                if full_vis: np.save(join(full_vis_path, k + '.npy'), v)
                self._vis_embed(v, outdir)
            elif k.endswith(('z')) and full_vis:  # HxWx3
                np.save(join(full_vis_path, k + '.npy'), v)
            elif k.endswith(('xyz',)):  # HxWx3
                if not exists(outdir): os.makedirs(outdir)
                np.save(join(outdir, k + '.npy'), v)
            # Normals
            elif k.endswith('normal'):
                v_ = (v + 1) / 2  # [-1, 1] to [0, 1]
                # v_ = v
                bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                img = imgutil.alpha_blend(v_, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            elif (not mode == 'render') and (not simp):
                img = v
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)

        if not simp:
            # Shortcircuit if testing because there will be no ground truth for
            # us to make .apng comparisons
            if mode in ('test', 'render'):
                # Write metadata that doesn't require ground truth (e.g., view name)
                metadata = {'id': id_}
                ioutil.write_json(metadata, join(outdir, 'metadata.json'))
                return

            # Write metadata (e.g., view name, PSNR, etc.)
            # official realization of NeRFactor
            psnr = self.psnr(img_dict['gt_rgb'], img_dict['pred_rgb']).tolist()
            metadata = {'id': id_, 'psnr': psnr}
            ioutil.write_json(metadata, join(outdir, 'metadata.json'))

    def _rgb2chromaticity(self, rgb):
        denom = tf.sqrt(tf.reduce_sum(tf.square(rgb), axis=-1, keepdims=True))
        return tf.math.divide_no_nan(rgb, denom)

    def _vis_embed(self, embed, outdir):
        embed_c = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
                   np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]),
                   np.array([128, 0, 0]), np.array([0, 128, 0]), np.array([0, 0, 128]),
                   np.array([128, 128, 0]), np.array([128, 0, 128]), np.array([0, 128, 128]),
                   np.array([255, 128, 128]), np.array([128, 255, 128]), np.array([128, 128, 255]),
                   np.array([255, 255, 128]), np.array([255, 128, 255]), np.array([128, 255, 255]), ]

        embed_map = np.zeros(embed.shape + (3,))
        for i in range(1, 19):
            embed_map[embed == i] = embed_c[i - 1]
        cv2.imwrite(join(outdir, 'embed_map.png'), embed_map)

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



