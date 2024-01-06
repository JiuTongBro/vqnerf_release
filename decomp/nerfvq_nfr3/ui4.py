import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import numpy as np
import cv2
import os
import time
from os.path import join
import json

import paramiko
from scp import SCPClient

class EditingWindow(QWidget):
    def __init__(self):
        super(EditingWindow, self).__init__()
        self.setFont(QFont("Roman times", 14))

        scene = 'drums_3072'
        self.local_folder = 'Your local cache folder'
        self.server_root = '/home/zhonghongliang/vqnfr_pro_release/decomp/nerfvq_nfr3/'

        self.server_path = self.server_root + 'output/train/' + scene + '_ref_nfr/lr5e-4/'
        self.defaulf_folder = self.local_folder + '/pd_comps'
        self.env_dir = self.local_folder + '/vis'

        time_out = 300.0
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname='Your host IP', port=22, username='Your user name', password='Your password',
                    compress=True)
        self.scpclient = SCPClient(ssh.get_transport(), socket_timeout=time_out)

        self.default_views = ['batch000000000', 'batch000000001', 'batch000000002']
        self.img_size = (280, 280)
        self.env_size = (200, 60)
        # self.img_size = (315, 420) # hwchair
        # self.img_size = (746, 420)
        # self.img_size = (631, 420) # ours
        # self.img_size = (682, 512) # dtu
        # self.img_size = (420, 420) # mat
        # self.img_size = (512, 512) # nerf
        # self.img_size = (560, 420) # tools2, redcar
        # self.img_size = (196, 420) # toyrabbit

        self.prop_map = {'rgb': 'pred_rgb.png', 'diff': 'pred_albedo.png', 'spec': 'pred_spec.png',
                    'rough': 'pred_rough.png', 'mat': 'mat', 'embed': 'embed_map.png',
                    'rgb_scale': 'pred_rgb.png', 'diff_scale': 'pred_albedo.png', 'spec_scale': 'pred_spec.png',
                    'xyz': 'xyz.npy', 'z': 'pred_z_bias.npy', 'ref': 'pred_ref.npy', }

        self.mat_db = {'Custom':{'diff': [-1, -1, -1], 'spec': [-1, -1, -1], 'rough': [-1]},
                  'BrightGold':{'diff': [0.1, 0.07, 0.02], 'spec': [0.85, 0.7, 0.35], 'rough': [0.35]},
                  'Iron':{'diff': [0.1, 0.2, 0.25], 'spec': [0.7, 0.7, 0.7], 'rough': [0.4]},
                  'Paper': {'diff': [0.4, 0.45, 0.35], 'spec': [0.7, 0.7, 0.7], 'rough': [0.8]},
                  'Rubber': {'diff': [0.01, 0.01, 0.01], 'spec': [0.05, 0.05, 0.05], 'rough': [0.6]},
                  'blue_china': {'diff': [0.35, 0.6, 0.55], 'spec': [0.1, 0.15, 0.15], 'rough': [0.45]},
                  'silver1': {'diff': [0.8, 0.9, 0.9], 'spec': [0.8, 0.9, 0.9], 'rough': [0.25]},
                  'Jade': {'diff': [0, 0.3, 0.2], 'spec': [0, 0.6, 0.5], 'rough': [0.5]},
                  'BluePaint': {'diff': [0, 0.3, 0.6], 'spec': [0, 0.2, 0.4], 'rough': [0.55]},
                  'PinkLeather': {'diff': [0.75, 0.05, 0.1], 'spec': [0.3, 0.2, 0.4], 'rough': [0.5]},
                  'PinkMetal': {'diff': [0.75, 0.05, 0.1], 'spec': [0.3, 0.2, 0.4], 'rough': [0.35]},
                  'RedMetal': {'diff': [0.3, 0.06, 0.1], 'spec': [0.6, 0.12, 0.2], 'rough': [0.35]},
                  'BlueMetal': {'diff': [0, 0.1, 0.2], 'spec': [0, 0.3, 0.6], 'rough': [0.3]},
                  'PurplePlastic': {'diff': [0.1, 0.05, 0.2], 'spec': [0.4, 0.02, 0.8], 'rough': [0.4]},
                  'Bronze': {'diff': [0., 0., 0.], 'spec': [0.83, 0.3, 0.14], 'rough': [0.28]},
                  'DarkGold': {'diff': [0., 0., 0.], 'spec': [0.31, 0.2, 0.03], 'rough': [0.25]},
                  'BrownMetal': {'diff': [0.06, 0.04, 0.03], 'spec': [0.28, 0.21, 0.16], 'rough': [0.3]},
                  'silver2': {'diff': [0.4, 0.45, 0.45], 'spec': [0.56, 0.63, 0.63], 'rough': [0.3]},
                  'RoseGold': {'diff': [0.04, 0.03, 0.03], 'spec': [0.43, 0.32, 0.28], 'rough': [0.3]}}

        self.envmaps = ['original', 'city', 'courtyard', 'forest', 'interior', 'night', 'studio', 'sunrise', 'sunset']

        self.selected_embeds = []
        self.dst_view1, self.dst_view2, self.dst_view3 = None, None, None

        self.InitUI()

    def InitUI(self):
        self.setMinimumSize(1000, self.img_size[1] + 650)

        self.h_bias = 110
        self.h_bias2 = 10
        self.w_bias = 330

        # Source Image Viewing
        self.src_label1 = QLabel(self)
        self.src_label1.setFixedSize(200, 30)
        self.src_label1.setText('Source View:')
        self.src_label1.move(95, 40)

        self.src_view_rgb = QLabel(self)
        self.src_view_rgb.setFixedSize(self.img_size[0], self.img_size[1])
        self.src_view_rgb.move(40, 80)
        self.src_view_rgb.setStyleSheet("QLabel{background:white;}")

        # Source Image Viewing
        self.src_label2 = QLabel(self)
        self.src_label2.setFixedSize(200, 30)
        self.src_label2.setText('Source Mask:')
        self.src_label2.move(95 + self.w_bias, 40)

        self.src_img = QLabel(self)
        self.src_img.setFixedSize(self.img_size[0], self.img_size[1])
        self.src_img.move(40 + self.w_bias, 80)
        self.src_img.setStyleSheet("QLabel{background:white;}")
        self.src_img.mousePressEvent = self.auto_select

        # Source Image Viewing
        self.src_label = QLabel(self)
        self.src_label.setFixedSize(200, 30)
        self.src_label.setText('Edited Mask:')
        self.src_label.move(95 + self.w_bias * 2, 40)

        self.edited_mask = QLabel(self)
        self.edited_mask.setFixedSize(self.img_size[0], self.img_size[1])
        self.edited_mask.move(40 + self.w_bias * 2, 80)
        self.edited_mask.setStyleSheet("QLabel{background:white;}")

        self.src_sel = QPushButton(self)
        self.src_sel.setText("Select")
        self.src_sel.move(130 + self.w_bias, 90 + self.img_size[1])
        self.src_sel.clicked.connect(self.sel_src)

        # Target View 1
        self.target_label1 = QLabel(self)
        self.target_label1.setFixedSize(200, 30)
        self.target_label1.setText('Target View 1:')
        self.target_label1.move(95, 40 + self.img_size[1] + self.h_bias)

        self.target_img1 = QLabel(self)
        self.target_img1.setFixedSize(self.img_size[0], self.img_size[1])
        self.target_img1.move(40, 80 + self.img_size[1] + self.h_bias)
        self.target_img1.setStyleSheet("QLabel{background:white;}")

        self.target_sel1 = QPushButton(self)
        self.target_sel1.setText("Select")
        self.target_sel1.move(130, 90 + self.img_size[1] + self.img_size[1] + self.h_bias)
        self.target_sel1.clicked.connect(self.sel_target1)

        # Target View 2
        self.target_label2 = QLabel(self)
        self.target_label2.setFixedSize(200, 30)
        self.target_label2.setText('Target View 2:')
        self.target_label2.move(95 + self.w_bias, 40 + self.img_size[1] + self.h_bias)

        self.target_img2 = QLabel(self)
        self.target_img2.setFixedSize(self.img_size[0], self.img_size[1])
        self.target_img2.move(40 + self.w_bias, 80 + self.img_size[1] + self.h_bias)
        self.target_img2.setStyleSheet("QLabel{background:white;}")

        self.target_sel2 = QPushButton(self)
        self.target_sel2.setText("Select")
        self.target_sel2.move(130 + self.w_bias, 90 + self.img_size[1] + self.img_size[1] + self.h_bias)
        self.target_sel2.clicked.connect(self.sel_target2)

        # Target View 3
        self.target_label3 = QLabel(self)
        self.target_label3.setFixedSize(200, 30)
        self.target_label3.setText('Target View 3:')
        self.target_label3.move(95 + self.w_bias * 2, 40 + self.img_size[1] + self.h_bias)

        self.target_img3 = QLabel(self)
        self.target_img3.setFixedSize(self.img_size[0], self.img_size[1])
        self.target_img3.move(40 + self.w_bias * 2, 80 + self.img_size[1] + self.h_bias)
        self.target_img3.setStyleSheet("QLabel{background:white;}")

        self.target_sel3 = QPushButton(self)
        self.target_sel3.setText("Select")
        self.target_sel3.move(130 + self.w_bias * 2, 90 + self.img_size[1] + self.img_size[1] + self.h_bias)
        self.target_sel3.clicked.connect(self.sel_target3)

        # --- auto ---
        self.mat_base = 50

        # Target Material:
        self.dst_label = QLabel(self)
        self.dst_label.setFixedSize(200, 30)
        self.dst_label.setText('Target Material:')
        self.dst_label.move(self.mat_base, 130 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.db_label = QLabel(self)
        self.db_label.setStyleSheet('font:16px;')
        self.db_label.setFixedSize(150, 20)
        self.db_label.setText('Databse:')
        self.db_label.move(self.mat_base, 170 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.db = QComboBox(self)
        self.db.setFixedSize(200, 30)
        self.db.move(self.mat_base + 70, 170 + self.img_size[1 ] * 2 + self.h_bias + self.h_bias2)

        for mat_name in self.mat_db.keys():
            self.db.addItem(mat_name)

        self.cus_label = QLabel(self)
        self.cus_label.setStyleSheet('font:16px;')
        self.cus_label.setFixedSize(150, 30)
        self.cus_label.setText('Custom:')
        self.cus_label.move(self.mat_base + 300, 170 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.diff_label = QLabel(self)
        self.diff_label.setStyleSheet('font:16px;')
        self.diff_label.setFixedSize(150, 30)
        self.diff_label.setText('diffuse:')
        self.diff_label.move(self.mat_base + 370, 140 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.diff = QLineEdit(self)
        self.diff.setStyleSheet('font:16px;')
        self.diff.setFixedSize(200, 20)
        self.diff.move(self.mat_base + 460, 145 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.spec_label = QLabel(self)
        self.spec_label.setStyleSheet('font:16px;')
        self.spec_label.setFixedSize(150, 30)
        self.spec_label.setText('specular:')
        self.spec_label.move(self.mat_base + 370, 170 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.spec = QLineEdit(self)
        self.spec.setStyleSheet('font:16px;')
        self.spec.setFixedSize(200, 20)
        self.spec.move(self.mat_base + 460, 175 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.rough_label = QLabel(self)
        self.rough_label.setStyleSheet('font:16px;')
        self.rough_label.setFixedSize(150, 30)
        self.rough_label.setText('roughness:')
        self.rough_label.move(self.mat_base + 370, 200 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.rough = QLineEdit(self)
        self.rough.setStyleSheet('font:16px;')
        self.rough.setFixedSize(200, 20)
        self.rough.move(self.mat_base + 460, 205 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        # Target Material:
        self.env_label = QLabel(self)
        self.env_label.setFixedSize(100, 30)
        self.env_label.setText('Env:')
        self.env_label.move(self.mat_base + 700, 130 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        self.env_db = QComboBox(self)
        self.env_db.setFixedSize(150, 30)
        self.env_db.move(self.mat_base + 760, 130 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)

        for env_name in self.envmaps:
            self.env_db.addItem(env_name)
        self.env_db.currentIndexChanged.connect(self.change_env)

        self.env_img = QLabel(self)
        self.env_img.setFixedSize(self.env_size[0], self.env_size[1])
        self.env_img.move(self.mat_base + 705, 170 + self.img_size[1] * 2 + self.h_bias + self.h_bias2)
        self.env_img.setStyleSheet("QLabel{background:white;}")

        # Run Selection
        self.auto_edit = QPushButton(self)
        self.auto_edit.setFixedSize(150, 30)
        self.auto_edit.setText("Edit!")
        self.auto_edit.move(150 + self.w_bias // 2, self.img_size[1] * 2 + self.h_bias + self.h_bias2 + 250)
        self.auto_edit.clicked.connect(self.edit)

        # Run Selection
        self.clear = QPushButton(self)
        self.clear.setFixedSize(150, 30)
        self.clear.setText("Clear")
        self.clear.move(360 + self.w_bias // 2, self.img_size[1] * 2 + self.h_bias + self.h_bias2 + 250)
        self.clear.clicked.connect(self.clear_edit)

        self.setWindowTitle("Editing Panel")
        self.show()

    def _normalize(self, arr):

        if arr.dtype not in (np.uint8, np.uint16):
            raise TypeError(arr.dtype)
        maxv = np.iinfo(arr.dtype).max
        arr_ = arr.astype(float)
        arr_ = arr_ / maxv
        return arr_

    def _cv2qt(self, img):
        img_height, img_width, channels = img.shape
        bytesPerLine = channels * img_width
        QImg = QImage(img.data, img_width, img_height, bytesPerLine, QImage.Format_RGB888)
        qt_img = QtGui.QPixmap.fromImage(QImg)
        return qt_img

    def openImage(self, path, dst_size=None):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2: img = img[..., None]
        if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)

        if dst_size is None:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
        else: img = cv2.resize(img, dst_size)

        qt_img = self._cv2qt(img)
        img = self._normalize(img)
        return img, qt_img

    def sel_src(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.src_view = os.path.dirname(imgName)
        self.img_dir = os.path.dirname(self.src_view)

        _, qt_img = self.openImage(imgName)
        self.src_view_rgb.setPixmap(qt_img)

        imgName = join(self.src_view, 'embed_map.png')
        self.src_rgb, qt_img = self.openImage(imgName)
        self.src_img.setPixmap(qt_img)

        self.mask_img, qt_img = self.openImage(imgName)
        self.edited_mask.setPixmap(qt_img)

    def sel_target1(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.dst_view1 = os.path.basename(os.path.dirname(imgName))
        self.target_rgb1, qt_img = self.openImage(imgName)
        self.target_img1.setPixmap(qt_img)

    def sel_target2(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.dst_view2 = os.path.basename(os.path.dirname(imgName))
        self.target_rgb2, qt_img = self.openImage(imgName)
        self.target_img2.setPixmap(qt_img)

    def sel_target3(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.dst_view3 = os.path.basename(os.path.dirname(imgName))
        self.target_rgb3, qt_img = self.openImage(imgName)
        self.target_img3.setPixmap(qt_img)

    def process_changing(self, cond_props,
                         edit_color=[1., 1., 1.]):

        # calculate a boundary
        prop_files = []
        for cond_prop in cond_props:
            prop_files.append(self.prop_map[cond_prop])

        cond = (self.src_rgb == self.selected_embeds[0])
        for i in range(1, len(self.selected_embeds)):
            cond = cond | (self.src_rgb == self.selected_embeds[i])
        cond_mask = np.all(cond, axis=-1)
        self.edited_color = np.array(edit_color)

        params = {}
        # for cond_prop, prop_file, k_bound in zip(cond_props, prop_files, k_bounds):
        for i in range(len(cond_props)):
            cond_prop = cond_props[i]
            params[cond_prop] = {'k': 0}

            samples = self.src_rgb[cond_mask]
            params[cond_prop]['edit'] = np.unique(samples, axis=0)

        return params

    def map_f(self, map_params, map_bg=False, eps=5e-6):

        dst, cond_flags, cond_bounds, conds_op, auto_params = map_params

        sel_dir = join(os.path.dirname(self.img_dir), 'auto_sel', 'edited')
        if not os.path.exists(sel_dir): os.makedirs(sel_dir)

        with open(join(sel_dir, 'dst.json'), 'w') as f:
            json.dump(dst, f)

        if cond_flags is None and auto_params is None:
            print('Specific your changing!')
            return

        conds_k, conds_v = [], []
        if len(cond_flags)>0:
            for k, v in zip(cond_flags, cond_bounds):
                cond = {}
                cond['min'] = np.array(v[0])
                cond['max'] = np.array(v[1])
                conds_k.append(k)
                conds_v.append(cond)

        view_list = os.listdir(self.img_dir)
        for v_dir in view_list:
            if not (v_dir[:5] == 'batch' or v_dir[:4] == 'test'): continue
            params = {}
            prop_files = [self.prop_map[k] for k in conds_k]
            for i in range(len(conds_k)):
                cond_prop = conds_k[i]
                prop_file = prop_files[i]
                if cond_prop == 'mat':
                    d = cv2.imread(join(self.src_view, 'pred_albedo.png'), -1)[..., [2, 1, 0]]
                    g = cv2.imread(join(self.src_view, 'pred_spec.png'), -1)[..., [2, 1, 0]]
                    r = cv2.imread(join(self.src_view, 'pred_rough.png'), -1)
                    if r.shape[-1] == 3: r = np.mean(r, axis=-1)
                    cond_src = np.concatenate([d, g, r[..., None]], axis=-1)
                    cond_src = cv2.resize(cond_src, self.img_size, interpolation=cv2.INTER_AREA)
                elif cond_prop[-5:] == 'scale':
                    cond_src, _ = self.openImage(join(self.img_dir, v_dir, prop_file))
                    cond_src = cond_src[..., :2] / (cond_src[..., 2:] + eps)
                elif prop_file[-4:] == '.npy':
                    prop_path = join(self.img_dir, v_dir, prop_file)
                    if prop_file == 'xyz.npy' and not os.path.exists(prop_path):
                        prop_path = prop_path.replace('xyz.npy', 'pred_xyz.npy')
                    cond_src = np.load(prop_path)
                    cond_src = cv2.resize(cond_src, self.img_size, interpolation=cv2.INTER_AREA)
                else:
                    cond_src, _ = self.openImage(join(self.img_dir, v_dir, prop_file))
                params[cond_prop] = cond_src

            conds_all = []
            for i in range(len(conds_k)):
                k = conds_k[i]
                v = conds_v[i]
                conds_all.append((params[k] > v['min']) & (params[k] < v['max']))

            if 'embed' in auto_params.keys():
                params_embed, _ = self.openImage(join(self.img_dir, v_dir, 'embed_map.png')) # h,w,3
                intend_material = (auto_params['embed']['edit']) [:, None, None, :] # n,1,1,3
                equal = params_embed[None, ...] - intend_material # n,h,w,3
                in_change = np.any(np.all(equal==0, axis=-1)==True, axis=0)
                conds_all.append(in_change[..., None])

            cond_map = np.all(conds_all[0], axis=-1)
            for i in range(1, len(cond_flags)):
                if conds_op[i - 1] == 'or':
                    cond_map = cond_map | (np.all(conds_all[i], axis=-1))
                else:
                    cond_map = cond_map & (np.all(conds_all[i], axis=-1))
            for i in range(len(cond_flags), len(conds_all)):
                cond_map = cond_map & (np.all(conds_all[i], axis=-1))
            cond_map = np.repeat(cond_map[..., None], 3, axis=-1)
            np.save(join(sel_dir, v_dir+'.npy'), cond_map)

        edit_mask = np.load(join(sel_dir, os.path.basename(self.src_view)+'.npy'))
        edit_mask = (np.where(edit_mask, 1., 0.) * 255).astype(np.uint8)
        cv2.imwrite(join(sel_dir, 'edit_mask.png'), edit_mask)

        # edited_mask
        dst_map = np.load(join(sel_dir, os.path.basename(self.src_view)+'.npy'))
        replaced_rgb = np.where(dst_map, self.edited_color, self.mask_img)
        replaced_rgb = (replaced_rgb * 255).astype(np.uint8)
        qt_img = self._cv2qt(replaced_rgb)
        self.edited_mask.setPixmap(qt_img)

        # target 1
        if self.dst_view1 is not None:
            dst_map = np.load(join(sel_dir, self.dst_view1+'.npy'))
            replaced_rgb = np.where(dst_map, self.edited_color, self.target_rgb1)
            replaced_rgb = (replaced_rgb * 255).astype(np.uint8)
            qt_img = self._cv2qt(replaced_rgb)
            self.target_img1.setPixmap(qt_img)

        # target 2
        if self.dst_view2 is not None:
            dst_map = np.load(join(sel_dir, self.dst_view2+'.npy'))
            replaced_rgb = np.where(dst_map, self.edited_color, self.target_rgb2)
            replaced_rgb = (replaced_rgb * 255).astype(np.uint8)
            qt_img = self._cv2qt(replaced_rgb)
            self.target_img2.setPixmap(qt_img)

        # target 3
        if self.dst_view3 is not None:
            dst_map = np.load(join(sel_dir, self.dst_view3+'.npy'))
            replaced_rgb = np.where(dst_map, self.edited_color, self.target_rgb3)
            replaced_rgb = (replaced_rgb * 255).astype(np.uint8)
            qt_img = self._cv2qt(replaced_rgb)
            self.target_img3.setPixmap(qt_img)

    # conds: rgb(3), xyz(3), diff(3), spec(3), rough(1), rgb_scale(2), diff_scale(2), spec_scale(2), z(256), ref(256)
    # params: rgb, xyz, albedo, spec, rough, rgb_scale(2), diff_scale(2), spec_scale(2), z_bias, rough; in model
    def auto_select(self, event):
        if self.src_rgb is None: return

        x = event.pos().x()
        y = event.pos().y()
        self.selected_embeds.append(self.src_rgb[y, x])  # reverse h and w

        cond_props = ['embed']

        cond_bounds = []
        cond_flags, conds_op, bounds_min, bounds_max = [],[],[],[]

        for i in range(len(bounds_min)):
            cond_bounds.append([bounds_min[i],bounds_max[i]])

        auto_params = self.process_changing(cond_props)

        dst_db = self.db.currentText()
        if dst_db == 'Custom':
            dst_diff = self.diff.text().split(',')
            dst_diff = list(map(float, dst_diff))
            dst_spec = self.spec.text().split(',')
            dst_spec = list(map(float, dst_spec))
            dst_rough = self.rough.text().split(',')
            dst_rough = list(map(float, dst_rough))
            dst = {'diff': dst_diff, 'spec': dst_spec, 'rough': dst_rough, }
        else: dst = self.mat_db[dst_db]

        map_params = [dst, cond_flags, cond_bounds, conds_op, auto_params]
        self.map_f(map_params)

    def edit(self, event):

        sel_dir = join(os.path.dirname(self.img_dir), 'auto_sel', 'edited')
        if not os.path.exists(sel_dir): os.makedirs(sel_dir)

        env = self.env_db.currentText()
        illum = {'env': env}
        with open(join(sel_dir, 'illum.json'), 'w') as f:
            json.dump(illum, f)

        sel_path = join(os.path.dirname(self.img_dir), 'status.json')

        local_mask_dir = join(os.path.dirname(self.img_dir), 'auto_sel', 'edited')
        remote_mask_dir = join(self.server_path, 'edited')
        files = os.listdir(local_mask_dir)
        for file in files:
            local_file = join(local_mask_dir, file)
            self.scpclient.put(local_file, remote_mask_dir)  ###

        status_json = {'status': 'uploaded'}
        with open(sel_path, 'w') as f:
            json.dump(status_json, f)

        server_status_path = self.server_root + 'status/status.json'
        self.scpclient.put(sel_path, server_status_path)  ###

        print('Upload Finished. Waiting...')
        while True:
            time.sleep(5)

            sel_path = join(os.path.dirname(self.img_dir), 'status.json')
            server_status_path = self.server_root + 'status/status.json'
            self.scpclient.get(server_status_path, os.path.dirname(self.img_dir))  ###
            with open(sel_path) as f: status_json = json.load(f)

            if status_json['status'] == 'finished':
                self.scpclient.get(self.server_path + 'pd_edited', os.path.dirname(self.img_dir), recursive=True)  ###
                print('Editing Finished.')
                break

        status_json = {'status': 'waiting'}
        with open(sel_path, 'w') as f:
            json.dump(status_json, f)

        server_status_path = self.server_root + 'status/status.json'
        self.scpclient.put(sel_path, server_status_path)  ###

        edited_dir = join(os.path.dirname(self.img_dir), 'pd_edited')

        # target 1
        if self.dst_view1 is not None:
            target_path = join(edited_dir, self.dst_view1, 'pred_rgb.png')
        else: target_path = join(edited_dir, self.default_views[0], 'pred_rgb.png')
        _, qt_img = self.openImage(target_path)
        self.target_img1.setPixmap(qt_img)

        # target 2
        if self.dst_view2 is not None:
            target_path = join(edited_dir, self.dst_view2, 'pred_rgb.png')
        else: target_path = join(edited_dir, self.default_views[1], 'pred_rgb.png')
        _, qt_img = self.openImage(target_path)
        self.target_img2.setPixmap(qt_img)

        # target 3
        if self.dst_view3 is not None:
            target_path = join(edited_dir, self.dst_view3, 'pred_rgb.png')
        else: target_path = join(edited_dir, self.default_views[2], 'pred_rgb.png')
        _, qt_img = self.openImage(target_path)
        self.target_img3.setPixmap(qt_img)

    def clear_edit(self, event):

        self.selected_embeds = []

        imgName = join(self.src_view, 'pred_rgb.png')
        _, qt_img = self.openImage(imgName)
        self.src_view_rgb.setPixmap(qt_img)

        imgName = join(self.src_view, 'embed_map.png')
        _, qt_img = self.openImage(imgName)
        self.src_img.setPixmap(qt_img)

        self.mask_img, qt_img = self.openImage(imgName)
        self.edited_mask.setPixmap(qt_img)

        self.target_img1.clear()
        self.target_img2.clear()
        self.target_img3.clear()

    def change_env(self, event):
        dst_env = self.env_db.currentText()
        if not dst_env == 'original':
            imgName = join(self.env_dir, dst_env+'.png')
            _, qt_img = self.openImage(imgName, dst_size=self.env_size)
            self.env_img.setPixmap(qt_img)
        else: self.env_img.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = EditingWindow()
    sys.exit(app.exec_())
