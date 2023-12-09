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
            self.render_cameras_name = 'train.json'
            self.object_cameras_name = 'train.json'
            prefix = 'train_*'
        else:
            self.render_cameras_name = 'val.json'
            self.object_cameras_name = 'val.json'
            prefix = 'val_*'

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        cam_dir=os.path.join(self.data_dir, self.render_cameras_name)
        with open(cam_dir) as f: camera_dict = json.load(f)
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, prefix)))
        self.n_images = len(self.images_lis)

        imgs_input=[self._read_rgba(os.path.join(im_name, 'rgba.png')) for im_name in self.images_lis]

        new_h = conf.get_float('new_h', default=0)
        if new_h>0:
            h, w = imgs_input[0].shape[0], imgs_input[0].shape[1]
            self.k = new_h / h
            new_w = w * self.k
            images_rgba=[cv.resize(im, (int(new_w), int(new_h))) for im in imgs_input]
        else:
            images_rgba = imgs_input
            self.k = 1.

        self.images_np = np.stack([im[...,:3] for im in images_rgba]) / 255.0
        self.masks_np = np.stack([np.repeat(im[...,3:],3,axis=-1) for im in images_rgba]) / 255.0

        # c2w in nerf data
        self.pose_all, self.intrinsics_all, self.scale_mats_np = [], [], []
        for idx in range(self.n_images):
            raw_scale = np.array(camera_dict['scale_mat'][idx])
            raw_world = np.array(camera_dict['world_mat'][idx])

            scaled_projection = (raw_world @ raw_scale)[0:3, 0:4]
            intrinsic, pose = self.decompose_projection_matrix(scaled_projection)
            intrinsic[:2, :3] = intrinsic[:2, :3] * self.k

            self.scale_mats_np.append(raw_scale.astype(np.float32))
            self.pose_all.append(torch.from_numpy(pose).float())
            self.intrinsics_all.append(torch.from_numpy(intrinsic).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]

        self.H, self.W = self.images.shape[1], self.images.shape[2]
        print('hw: ',self.H, self.W)
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]

        self.image_pixels = self.H * self.W

        self.max_radius = 1.
        self.near, self.far = self.compute_near_far()
        print('near: ', self.near, '  far: ', self.far, '  radius: ', self.max_radius)

        eps = 0.01
        min_bound = - (self.max_radius + eps)
        max_bound = self.max_radius + eps
        object_bbox_min = np.array([min_bound, min_bound, min_bound, 1.0])
        object_bbox_max = np.array([max_bound, max_bound, max_bound, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = self.scale_mats_np[0]
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def compute_near_far(self):
        nears, fars = [], []
        np_ones = np.array([[1.]])
        for pose_torch in self.pose_all:
            pose = pose_torch.detach().cpu().numpy()
            cam_pos = pose[:3, 3:] # 3,1
            o2cam = cam_pos / np.linalg.norm(cam_pos, ord=2, axis=0, keepdims=True)
            n_p = o2cam * self.max_radius # 3,1
            f_p = - n_p # 3,1
            n_p = np.concatenate([n_p, np_ones], axis=0)
            f_p = np.concatenate([f_p, np_ones], axis=0)
            w2c = np.linalg.inv(pose) # 3,4

            nears.append((w2c @ n_p)[2, 0])
            fars.append((w2c @ f_p)[2, 0])

        return np.min(nears), np.max(fars)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        rgba = self._read_rgba(os.path.join(self.images_lis[idx], 'rgba.png'))
        img = cv.resize(rgba[...,:3], (self.W // resolution_level, self.H // resolution_level))
        return img.clip(0, 255)

    def _read_rgba(self, path, longint=False):
        if longint:
            uint16_img = cv.imread(path, -1)
            uint8_img = (uint16_img // 256).astype(np.uint8)
        else: uint8_img = cv.imread(path, -1)
        return uint8_img.clip(0, 255)

    def decompose_projection_matrix(self, P):
        ''' Decompose intrincis and extrinsics from projection matrix (for Numpt object) '''
        # For Numpy object

        out = cv.decomposeProjectionMatrix(P)
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


