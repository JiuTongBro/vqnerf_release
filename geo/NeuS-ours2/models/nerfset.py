import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import json
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

class Dataset:
    def __init__(self, conf, is_train=True):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        if is_train:
            self.render_cameras_name = 'transforms_train.json'
            self.object_cameras_name = 'transforms_train.json'
            prefix = 'train_*'
        else:
            self.render_cameras_name = 'transforms_val.json'
            self.object_cameras_name = 'transforms_val.json'
            prefix = 'val_*'

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.near = conf.get_float('near', default=2.)
        self.far = conf.get_float('far', default=6.)
        self.longint = conf.get_bool('longint', default=True)

        cam_dir=os.path.join(self.data_dir, self.render_cameras_name)
        with open(cam_dir) as f: camera_dict = json.load(f)
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, prefix)))
        self.n_images = len(self.images_lis)


        if 'cx' in camera_dict.keys():
            self.cx, self.cy = camera_dict['cx'], camera_dict['cy']
        else: self.cx, self.cy = None, None


        imgs_input=[self._read_rgba(os.path.join(im_name, 'rgba.png'), longint=self.longint) for im_name in self.images_lis]

        new_h = conf.get_float('new_h', default=0)
        if new_h>0:
            h, w = imgs_input[0].shape[0], imgs_input[0].shape[1]
            k = new_h / h
            new_w = w * k
            images_rgba=[cv.resize(im, (int(new_w), int(new_h))) for im in imgs_input]
            if self.cx is not None:
                self.cx, self.cy = self.cx * k, self.cy * k
        else: images_rgba = imgs_input

        self.images_np = np.stack([im[...,:3] for im in images_rgba]) / 255.0
        #self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([np.repeat(im[...,3:],3,axis=-1) for im in images_rgba]) / 255.0

        # c2w in nerf data
        self.pose_all = []
        for idx in range(self.n_images):
            pose_mat = camera_dict['frames'][idx]['transform_matrix']
            if isinstance(pose_mat, str):
                pose_ = np.array([float(x) for x in pose_mat.split(',')]).reshape(4, 4).astype(np.float32)
            else: pose_ = np.array(pose_mat).astype(np.float32)
            self.pose_all.append(torch.from_numpy(pose_.reshape(4, 4).astype(np.float32)).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]

        cam_angle_x = camera_dict['camera_angle_x']
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.focal = .5 * self.W / np.tan(.5 * cam_angle_x)
        print('hwf: ',self.H, self.W, self.focal)
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        self.image_pixels = self.H * self.W

        self.max_radius = self._get_radius()
        print('near: ', self.near, '  far: ', self.far, '  radius: ', self.max_radius, '  longint: ', self.longint)
        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = np.array([-1.1, -1.1, -1.1]) * self.max_radius
        self.object_bbox_max = np.array([1.1,  1.1,  1.1]) * self.max_radius

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, gen_mask=False):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)

        if self.cx is None: cx, cy = self.W // 2, self.H // 2
        else: cx, cy = int(self.cx), int(self.cy)
        p = torch.stack([(pixels_x - cx) / self.focal, -(pixels_y - cy) / self.focal,
                         -torch.ones_like(pixels_y)], dim=-1)  # W, H, 3

        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        if gen_mask:
            mask = self.masks[img_idx, :, :, :1].cuda()
            return rays_o.transpose(0, 1), rays_v.transpose(0, 1), mask
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1) # H, W, 3

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3

        if self.cx is None: cx, cy = self.W // 2, self.H // 2
        else: cx, cy = int(self.cx), int(self.cy)
        p = torch.stack([(pixels_x - cx) / self.focal, -(pixels_y - cy) / self.focal,
                         -torch.ones_like(pixels_y)], dim=-1)  # batch_size, 3

        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d):
        temp = torch.sum(rays_d**2, dim=-1, keepdim=True)
        near = torch.ones_like(temp) * self.near
        far = torch.ones_like(temp) * self.far
        return near, far

    def _get_radius(self):
        bd = np.array([[0., 0.], [0., 0.], [-self.near, -self.far], [1., 1.], ])
        c2ws = self.pose_all.cpu().numpy()
        max_radius = 0
        for c2w in c2ws:
            r = np.max(np.sqrt(np.sum(np.square((c2w @ bd)[:3, :]), axis=0)))
            if r > max_radius: max_radius = r
        return max_radius

    def image_at(self, idx, resolution_level):
        rgba = self._read_rgba(os.path.join(self.images_lis[idx], 'rgba.png'), longint=self.longint)
        img = cv.resize(rgba[...,:3], (self.W // resolution_level, self.H // resolution_level))
        return img.clip(0, 255)

    def _read_rgba(self, path, longint=True):
        if longint:
            uint16_img = cv.imread(path, -1)
            uint8_img = (uint16_img // 256).astype(np.uint8)
        else: uint8_img = cv.imread(path, -1)
        return uint8_img.clip(0, 255)

