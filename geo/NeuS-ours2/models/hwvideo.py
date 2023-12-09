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
    def __init__(self, train_set):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = train_set.conf

        self.data_dir = train_set.data_dir
        print(self.data_dir)
        self.render_cameras_name = 'transforms_test.json'

        self.near = train_set.near
        self.far = train_set.far
        self.longint = train_set.longint

        cam_dir = os.path.join(self.data_dir, self.render_cameras_name)
        if not os.path.exists(cam_dir):
            print('# Not Exist Test Json: ', cam_dir)
            data_root = os.path.dirname(os.path.dirname(self.data_dir))
            cam_dir = os.path.join(data_root, self.render_cameras_name)
        with open(cam_dir) as f: camera_dict = json.load(f)
        self.camera_dict = camera_dict
        self.n_images = len(camera_dict['frames'])

        self.cx, self.cy = train_set.cx, train_set.cy

        # c2w in nerf data
        self.pose_all = []
        for idx in range(self.n_images):
            pose_mat = camera_dict['frames'][idx]['transform_matrix']
            if isinstance(pose_mat, str):
                pose_ = np.array([float(x) for x in pose_mat.split(',')]).reshape(4, 4).astype(np.float32)
            else: pose_ = np.array(pose_mat).astype(np.float32)
            self.pose_all.append(torch.from_numpy(pose_.reshape(4, 4).astype(np.float32)).float())

        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        print(self.pose_all[0])

        self.H, self.W = train_set.H, train_set.W
        self.focal = train_set.focal
        print('hwf: ',self.H, self.W, self.focal)

        self.image_pixels = self.H * self.W
        self.max_radius = train_set.max_radius
        print('near: ', self.near, '  far: ', self.far, '  radius: ', self.max_radius, '  longint: ', self.longint)

        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = train_set.object_bbox_min
        self.object_bbox_max = train_set.object_bbox_max

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

    def near_far_from_sphere(self, rays_o, rays_d):
        temp = torch.sum(rays_d**2, dim=-1, keepdim=True)
        near = torch.ones_like(temp) * self.near
        far = torch.ones_like(temp) * self.far
        return near, far

