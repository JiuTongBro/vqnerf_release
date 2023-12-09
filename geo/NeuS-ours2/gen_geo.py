import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.nerfset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.util import gen_light_xyz
import math
from models.helpers import *

light_h = 16

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.scene_out_dir = self.conf['general.scene_out_dir']
        os.makedirs(self.scene_out_dir, exist_ok=True)
        self.train_data = Dataset(self.conf['dataset'])
        self.val_data = Dataset(self.conf['dataset'], is_train=False)
        self.n_train = self.train_data.n_images
        self.n_val = self.val_data.n_images
        self.iter_step = 0


        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.alpha_thres = self.conf.get_float('train.alpha_thres', default=0.5)
        print('# Pred Alpha Thres: ', self.alpha_thres)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def validate_image(self, idx=-1, resolution_level=-1, is_train=True, num_p=None, p_i=None, no_vis=False):
        if is_train:
            n_imgs = self.n_train
            dataset = self.train_data
            prefix = 'train_'
        else:
            n_imgs = self.n_val
            dataset = self.val_data
            prefix = 'val_'

        light_w = 2 * light_h
        lxyz, lareas = gen_light_xyz(light_h, light_w)
        lxyz = torch.from_numpy(lxyz.astype(np.float32)).cuda()
        lxyz_flat = lxyz.reshape((1, -1, 3))

        if num_p is None: frame_range = range(n_imgs)
        else:
            p_step = math.ceil(n_imgs / num_p)
            st, ed = p_i * p_step, (p_i + 1) * p_step
            # st, ed = 50 * (p_i + 1) - 4, 50 * (p_i + 1)
            frame_range = range(st, ed)

        # Process Geometry
        for idx in frame_range:
            if idx >= n_imgs:
                print('Skip Processing: ', p_i, idx)
                break

            print('Processing ', prefix + '{i:03d}'.format(i=idx), ' ...')
            view_dir = os.path.join(self.scene_out_dir, prefix + '{i:03d}'.format(i=idx))
            if not os.path.exists(view_dir): os.makedirs(view_dir, exist_ok=True)

            if self.check_finished(view_dir):
                print('Skip View: ', idx)
                continue

            print('Processing Geometry: ', idx)
            if is_train: alpha_thres = 0.5
            else: alpha_thres = self.alpha_thres
            surf, normal, mask = self.compute_geo(idx, resolution_level, dataset, view_dir, alpha_thres=alpha_thres)

            if not no_vis:
                surf = torch.from_numpy(surf[..., 0].astype(np.float32)).cuda()
                normal = torch.from_numpy(normal[..., 0].astype(np.float32)).cuda()
                mask = torch.from_numpy(mask[..., 0].astype(np.float32)).cuda()

                print('Processing Visibility: ', idx)
                # The geometry occlusion in CG data are more complex, introduce the visibility term for rendering
                # For training images, compute visibility on GT Mask, as the model will be trained on GT Mask
                if is_train:
                    self.compute_vis(idx, resolution_level, dataset, view_dir, lxyz_flat, surf, normal)
                else:
                    # not train and pd_alpha
                    self.compute_vis(idx, resolution_level, dataset, view_dir, lxyz_flat, surf, normal, mask)


    def compute_vis(self, idx, resolution_level, dataset, view_dir, lxyz_flat, surf, normal, mask=None, lpix_chunk=1):
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        if mask is None: rays_o, rays_d, mask = dataset.gen_rays_at(idx, resolution_level=resolution_level, gen_mask=True)
        else: rays_o, rays_d= dataset.gen_rays_at(idx, resolution_level=resolution_level)

        H, W, _ = rays_o.shape
        n_lights = lxyz_flat.shape[1]

        alpha = mask[..., 0] > 0.
        surf, normal = surf[alpha], normal[alpha]
        surf = surf.split(self.batch_size)
        normal = normal.split(self.batch_size)

        # print('Batch Nums: ', len(rays_o))
        batch_lvis = []
        for surf_batch, normal_batch in zip(surf, normal):
            lvis_hit = np.zeros((surf_batch.shape[0], n_lights), dtype=np.float32)  # (n_surf_pts, n_lights)

            # print('Start Another Batch...')
            for i in range(0, n_lights, lpix_chunk):
                end_i = min(n_lights, i + lpix_chunk)
                lxyz_chunk = lxyz_flat[:, i:end_i, :]  # (1, lpix_chunk, 3)

                # From surface to lights
                surf2l = lxyz_chunk - surf_batch[:, None, :]  # (n_surf_pts, lpix_chunk, 3)
                surf2l = surf2l / torch.linalg.norm(surf2l, ord=2, dim=-1, keepdim=True)  # (n_surf_pts, lpix_chunk, 3)
                surf2l_flat = surf2l.reshape((-1, 3))  # (n_surf_pts * lpix_chunk, 3), d

                surf_rep = surf_batch[:, None, :].repeat(1, surf2l.shape[1], 1)
                surf_flat = surf_rep.reshape((-1, 3))  # (n_surf_pts * lpix_chunk, 3), o

                # Save memory by ignoring back-lit points
                lcos = torch.einsum('ijk,ik->ij', surf2l, normal_batch)
                front_lit = lcos > 0  # (n_surf_pts, lpix_chunk)
                if torch.sum(front_lit) == 0:
                    # If there is no point being front lit, this visibility buffer is
                    # zero everywhere, so no need to update this slice
                    continue

                front_lit_flat = front_lit.reshape((-1,))  # (n_surf_pts * lpix_chunk)
                surf_flat_frontlit = surf_flat[front_lit_flat] # o
                surf2l_flat_frontlit = surf2l_flat[front_lit_flat] # d

                far, _ = self.intersect_circle(surf_flat_frontlit, surf2l_flat_frontlit, dataset.max_radius)
                n_far, n_near = far / 2., torch.ones_like(far) * 0.1
                near = torch.where(n_near<n_far, n_near, n_far)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(surf_flat_frontlit,
                                                  surf2l_flat_frontlit,
                                                  near,
                                                  far,
                                                  dataset.max_radius,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)

                occu = render_out['weight_sum'].detach().cpu().numpy()
                front_lit_full = np.zeros(lvis_hit.shape, dtype=bool)
                front_lit_full[:, i:end_i] = front_lit.detach().cpu().numpy()
                lvis_hit[front_lit_full] = 1. - occu[:, 0]
                del render_out

            batch_lvis.append(lvis_hit)

        lvis_hit = np.concatenate(batch_lvis, axis=0)

        # Put the light visibility values into the full tensor
        hit_map = alpha.detach().cpu().numpy().reshape((H, W, 1,))
        lvis = np.zeros(  # (imh, imw, n_lights)
            (H, W, n_lights,), dtype=np.float32)
        lvis[np.broadcast_to(hit_map, lvis.shape)] = lvis_hit.ravel()

        lvis_img = (np.mean(lvis, axis=-1, keepdims=True) * 256).clip(0, 255)
        cv.imwrite(os.path.join(view_dir, 'lvis.png'), lvis_img)
        np.save(os.path.join(view_dir, 'lvis.npy'), lvis)

    def compute_geo(self, idx, resolution_level, dataset, view_dir, alpha_thres=0.5):

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_surf = []
        out_mask = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

            near, far = dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              dataset.max_radius,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('weight_sum'):
                alpha_mask = render_out['weight_sum'].detach().cpu().numpy()
                alpha_mask = np.where(alpha_mask > alpha_thres, 1., 0.)
                out_mask.append(alpha_mask)
            if feasible('surf'):
                surf = render_out['surf'].detach().cpu().numpy()
                out_surf.append(surf)

                if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach().cpu().numpy() # n,3
                    normals = self._np_norm(normals, dim=-1)
                    rayo = rays_o_batch.detach().cpu().numpy()
                    normals = self.normal_correct(rayo, surf, normals)
                    out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        if len(out_surf) > 0:
            img_surf = np.concatenate(out_surf, axis=0).reshape([H, W, 3, -1])

        if len(out_mask) > 0:
            img_mask = (np.concatenate(out_mask, axis=0).reshape([H, W, 1, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot_normal = normal_img[:, :, None].reshape([H, W, 3, -1])
            rot_normal = rot_normal * (img_mask / 255.) + \
                         self._np_norm(np.ones_like(rot_normal), dim=-2) * (1. - img_mask / 255.)
            normal_img = (rot_normal * 128 + 128).clip(0, 255)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(view_dir, 'rgb.png'), img_fine[..., i])

            if len(out_surf) > 0:
                cv.imwrite(os.path.join(view_dir, 'xyz.png'), img_surf[..., i])
                np.save(os.path.join(view_dir, 'xyz.npy'), img_surf[..., i])

            if len(out_mask) > 0:
                cv.imwrite(os.path.join(view_dir, 'alpha.png'), img_mask[..., i])

            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(view_dir, 'normal.png'), normal_img[..., i])
                np.save(os.path.join(view_dir, 'normal.npy'), rot_normal[..., i])

        return img_surf, rot_normal, img_mask

    def intersect_circle(self, x, d, r, eps=1e-7):
        b = 2. * torch.sum(x * d, dim=-1)
        a = torch.sum(d * d, dim=-1)
        c = torch.sum(x * x, dim=-1) - r ** 2

        eps = torch.ones_like(a) * eps
        denom = torch.where(2 * a > eps, 2 * a, eps)
        t1 = (-b + torch.sqrt(torch.square(b) - 4. * a * c)) / denom
        t2 = (-b - torch.sqrt(torch.square(b) - 4. * a * c)) / denom
        t = torch.where(t1 > t2, t1, t2)

        return t[:, None], x + t[:, None] * d

    def normal_correct(self, rays_o, surf, normal):
        surf2c = rays_o - surf
        surf2c = surf2c / np.linalg.norm(surf2c, ord=2, axis=-1, keepdims=True)
        cos = np.sum(surf2c * normal, axis=-1, keepdims=True)
        normal = np.where(cos >= 0., normal, -normal)

        return normal

    def _np_norm(self, src, dim):
        r = np.sqrt(np.sum(np.square(src), axis=dim, keepdims=True))
        return src / r

    def check_finished(self, view_dir):
        view_files = ['lvis.npy', 'lvis.png', 'alpha.png',
                      'normal.npy', 'normal.png', 'rgb.png',
                      'xyz.npy', 'xyz.png']

        for f_name in view_files:
            f_path = os.path.join(view_dir, f_name)
            if not os.path.exists(f_path): return False

        return True


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--num_p', type=int, default=None)
    parser.add_argument('--p_i', type=int, default=None)
    parser.add_argument('--only_val', default=False, action="store_true")
    parser.add_argument('--no_vis', default=False, action="store_true")

    args = parser.parse_args()

    args.is_continue = True
    if args.conf is None:
        args.conf = conf_dict[args.case]
    if not args.case in cg_data_list:
        args.no_vis = True

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if not args.only_val:
        runner.validate_image(is_train=True, num_p=args.num_p, p_i=args.p_i, no_vis=args.no_vis)
    runner.validate_image(is_train=False, num_p=args.num_p, p_i=args.p_i, no_vis=args.no_vis)
