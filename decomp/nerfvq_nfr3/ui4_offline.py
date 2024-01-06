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

class EditingWindow(QWidget):
    def __init__(self):
        super(EditingWindow, self).__init__()
        self.setFont(QFont("Roman times", 14))
        self.defaulf_folder = 'D:/1/tvcg/src'

        self.img_size = (315, 420) # hwchair
        # self.img_size = (746, 420) # hw
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
                  'Gold':{'diff': [0.1, 0.07, 0.02], 'spec': [0.85, 0.7, 0.35], 'rough': [0.35]},
                  'Iron':{'diff': [0.1, 0.2, 0.25], 'spec': [0.7, 0.7, 0.7], 'rough': [0.4]},
                  'Paper': {'diff': [0.4, 0.45, 0.35], 'spec': [0.7, 0.7, 0.7], 'rough': [0.8]},
                  'Rubber': {'diff': [0.01, 0.01, 0.01], 'spec': [0.05, 0.05, 0.05], 'rough': [0.6]},
                  'blue_china': {'diff': [0.35, 0.6, 0.55], 'spec': [0.1, 0.15, 0.15], 'rough': [0.45]},
                  'silver1': {'diff': [0.8, 0.9, 0.9], 'spec': [0.8, 0.9, 0.9], 'rough': [0.25]},
                  'Jade': {'diff': [0, 0.3, 0.2], 'spec': [0, 0.6, 0.5], 'rough': [0.5]},
                  'BlueDiff': {'diff': [0, 0.3, 0.6], 'spec': [0, 0.2, 0.4], 'rough': [0.55]},
                  'PinkLeather': {'diff': [0.75, 0.05, 0.1], 'spec': [0.3, 0.2, 0.4], 'rough': [0.5]},
                  'PinkIron': {'diff': [0.75, 0.05, 0.1], 'spec': [0.3, 0.2, 0.4], 'rough': [0.35]},
                  'RedIron': {'diff': [0.3, 0.06, 0.1], 'spec': [0.6, 0.12, 0.2], 'rough': [0.35]},
                  'BlueIron': {'diff': [0, 0.1, 0.2], 'spec': [0, 0.3, 0.6], 'rough': [0.3]},
                  'PurplePlastic': {'diff': [0.1, 0.05, 0.2], 'spec': [0.4, 0.02, 0.8], 'rough': [0.4]},
                  'Bronze': {'diff': [0., 0., 0.], 'spec': [0.83, 0.3, 0.14], 'rough': [0.28]},
                  'YellowMetal': {'diff': [0., 0., 0.], 'spec': [0.31, 0.2, 0.03], 'rough': [0.25]},
                  'BrownMetal': {'diff': [0.06, 0.04, 0.03], 'spec': [0.28, 0.21, 0.16], 'rough': [0.3]},
                  'silver2': {'diff': [0.4, 0.45, 0.45], 'spec': [0.56, 0.63, 0.63], 'rough': [0.3]},
                  'RoseGold': {'diff': [0.04, 0.03, 0.03], 'spec': [0.43, 0.32, 0.28], 'rough': [0.3]}}

        self.selected_embeds = []
        self.InitUI()

    def InitUI(self):
        self.setMinimumSize(1800, self.img_size[1] + 450)

        # Source Image Viewing
        self.src_label = QLabel(self)
        self.src_label.setFixedSize(200, 30)
        self.src_label.setText('Source Image:')
        self.src_label.move(230, 40)

        self.src_img = QLabel(self)
        self.src_img.setFixedSize(self.img_size[0], self.img_size[1])
        self.src_img.move(50, 80)
        self.src_img.setStyleSheet("QLabel{background:white;}")
        self.src_img.mousePressEvent = self.auto_select

        self.src_sel = QPushButton(self)
        self.src_sel.setText("Select")
        self.src_sel.move(240, 90 + self.img_size[1])
        self.src_sel.clicked.connect(self.sel_src)

        # Target View
        self.target_label = QLabel(self)
        self.target_label.setFixedSize(200, 30)
        self.target_label.setText('Target Image:')
        self.target_label.move(1020, 40)

        self.target_img = QLabel(self)
        self.target_img.setFixedSize(self.img_size[0], self.img_size[1])
        self.target_img.move(1020, 80)
        self.target_img.setStyleSheet("QLabel{background:white;}")

        self.target_sel = QPushButton(self)
        self.target_sel.setText("Select")
        self.target_sel.move(1020, 90 + self.img_size[1])
        self.target_sel.clicked.connect(self.sel_target)

        # --- auto ---
        self.flag_label = QLabel(self)
        self.flag_label.setStyleSheet('font:16px;')
        self.flag_label.setFixedSize(150, 30)
        self.flag_label.setText('Manual_Cond:')
        self.flag_label.move(120, 250 + self.img_size[1])

        self.flag = QLineEdit(self)
        self.flag.setStyleSheet('font:16px;')
        self.flag.setFixedSize(550, 20)
        self.flag.move(250, 255 + self.img_size[1])

        self.op_label = QLabel(self)
        self.op_label.setStyleSheet('font:16px;')
        self.op_label.setFixedSize(120, 30)
        self.op_label.setText('Manual_Op:')
        self.op_label.move(120, 280 + self.img_size[1])

        self.op = QLineEdit(self)
        self.op.setStyleSheet('font:16px;')
        self.op.setFixedSize(550, 20)
        self.op.move(250, 285 + self.img_size[1])

        self.min_label = QLabel(self)
        self.min_label.setStyleSheet('font:16px;')
        self.min_label.setFixedSize(120, 30)
        self.min_label.setText('Bound_Min:')
        self.min_label.move(120, 310 + self.img_size[1])

        self.bound_min = QLineEdit(self)
        self.bound_min.setStyleSheet('font:16px;')
        self.bound_min.setFixedSize(550, 20)
        self.bound_min.move(250, 315 + self.img_size[1])

        self.max_label = QLabel(self)
        self.max_label.setStyleSheet('font:16px;')
        self.max_label.setFixedSize(120, 30)
        self.max_label.setText('Bound_Max:')
        self.max_label.move(120, 340 + self.img_size[1])

        self.bound_max = QLineEdit(self)
        self.bound_max.setStyleSheet('font:16px;')
        self.bound_max.setFixedSize(550, 20)
        self.bound_max.move(250, 345 + self.img_size[1])

        # Target Material:
        self.dst_label = QLabel(self)
        self.dst_label.setFixedSize(200, 30)
        self.dst_label.setText('Target Material:')
        self.dst_label.move(120, 130 + self.img_size[1])

        self.db_label = QLabel(self)
        self.db_label.setStyleSheet('font:16px;')
        self.db_label.setFixedSize(150, 20)
        self.db_label.setText('Databse:')
        self.db_label.move(120, 170 + self.img_size[1])

        self.db = QComboBox(self)
        self.db.setFixedSize(150, 30)
        self.db.move(210, 170 + self.img_size[1])

        for mat_name in self.mat_db.keys():
            self.db.addItem(mat_name)

        self.cus_label = QLabel(self)
        self.cus_label.setStyleSheet('font:16px;')
        self.cus_label.setFixedSize(150, 30)
        self.cus_label.setText('Custom:')
        self.cus_label.move(400, 170 + self.img_size[1])

        self.diff_label = QLabel(self)
        self.diff_label.setStyleSheet('font:16px;')
        self.diff_label.setFixedSize(150, 30)
        self.diff_label.setText('diffuse:')
        self.diff_label.move(470, 140 + self.img_size[1])

        self.diff = QLineEdit(self)
        self.diff.setStyleSheet('font:16px;')
        self.diff.setFixedSize(200, 20)
        self.diff.move(560, 145 + self.img_size[1])

        self.spec_label = QLabel(self)
        self.spec_label.setStyleSheet('font:16px;')
        self.spec_label.setFixedSize(150, 30)
        self.spec_label.setText('metalness:')
        self.spec_label.move(470, 170 + self.img_size[1])

        self.spec = QLineEdit(self)
        self.spec.setStyleSheet('font:16px;')
        self.spec.setFixedSize(200, 20)
        self.spec.move(560, 175 + self.img_size[1])

        self.rough_label = QLabel(self)
        self.rough_label.setStyleSheet('font:16px;')
        self.rough_label.setFixedSize(150, 30)
        self.rough_label.setText('roughness:')
        self.rough_label.move(470, 200 + self.img_size[1])

        self.rough = QLineEdit(self)
        self.rough.setStyleSheet('font:16px;')
        self.rough.setFixedSize(200, 20)
        self.rough.move(560, 205 + self.img_size[1])

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

    def openImage(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2: img = img[..., None]
        if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
        qt_img = self._cv2qt(img)
        img = self._normalize(img)
        return img, qt_img

    def sel_src(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.src_view = os.path.dirname(imgName)
        self.img_dir = os.path.dirname(self.src_view)
        self.src_rgb, qt_img = self.openImage(imgName)
        self.src_img.setPixmap(qt_img)

    def sel_target(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Select Image", self.defaulf_folder, "All Files(*)")
        self.dst_view = os.path.basename(os.path.dirname(imgName))
        self.target_rgb, qt_img = self.openImage(imgName)
        self.target_img.setPixmap(qt_img)

    def process_changing(self, cond_props,
                         edit_color=[1., 1., 1.]):

        # calculate a boundary
        prop_files = []
        for cond_prop in cond_props:
            prop_files.append(self.prop_map[cond_prop])

        cond = np.all(self.src_rgb == self.selected_embeds[0], axis=-1)
        for i in range(1, len(self.selected_embeds)):
            cond = cond | np.all(self.src_rgb == self.selected_embeds[i], axis=-1)
        cond_mask = cond
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

        sel_dir = join(os.path.dirname(self.img_dir), 'auto_sel', os.path.basename(self.img_dir))
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
                elif cond_prop[-5:] == 'scale':
                    cond_src, _ = self.openImage(join(self.img_dir, v_dir, prop_file))
                    cond_src = cond_src[..., :2] / (cond_src[..., 2:] + eps)
                elif prop_file[-4:] == '.npy':
                    prop_path = join(self.img_dir, v_dir, prop_file)
                    if prop_file == 'xyz.npy' and not os.path.exists(prop_path):
                        prop_path = prop_path.replace('xyz.npy', 'pred_xyz.npy')
                    cond_src = np.load(prop_path)
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


        dst_map = np.load(join(sel_dir, self.dst_view+'.npy'))
        replaced_rgb = np.where(dst_map, self.edited_color, self.target_rgb)
        replaced_rgb = (replaced_rgb * 255).astype(np.uint8)
        qt_img = self._cv2qt(replaced_rgb)
        self.target_img.setPixmap(qt_img)

    # conds: rgb(3), xyz(3), diff(3), spec(3), rough(1), rgb_scale(2), diff_scale(2), spec_scale(2), z(256), ref(256)
    # params: rgb, xyz, albedo, spec, rough, rgb_scale(2), diff_scale(2), spec_scale(2), z_bias, rough; in model
    def auto_select(self, event):
        if self.src_rgb is None: return

        x = event.pos().x()
        y = event.pos().y()
        self.selected_embeds.append(self.src_rgb[y, x])  # reverse h and w

        cond_props = ['embed']

        cond_flags = self.flag.text().split(',')
        conds_op = self.op.text().split(',')

        min_text = self.bound_min.text().split(';')
        max_text = self.bound_max.text().split(';')

        cond_bounds = []
        if not cond_flags[0] == '':
            bounds_min = [list(map(float, e.split(','))) for e in min_text]
            bounds_max = [list(map(float, e.split(','))) for e in max_text]
        else:
            cond_flags, conds_op, bounds_min, bounds_max = [],[],[],[]

        for i in range(len(bounds_min)):
            cond_bounds.append([bounds_min[i],bounds_max[i]])

        auto_params = self.process_changing(cond_props)

        dst_db = self.db.currentText()
        if dst_db == 'Custom':
            dst_diff = self.diff.text().split(',')
            if dst_diff[0] == '': dst_diff = [-1, -1, -1]
            else: dst_diff = list(map(float, dst_diff))

            dst_spec = self.spec.text().split(',')
            if dst_spec[0] == '': dst_spec = [-1, -1, -1]
            else: dst_spec = list(map(float, dst_spec))

            dst_rough = self.rough.text().split(',')
            if dst_rough[0] == '': dst_rough = [-1]
            else: dst_rough = list(map(float, dst_rough))
            dst = {'diff': dst_diff, 'spec': dst_spec, 'rough': dst_rough, }
        else: dst = self.mat_db[dst_db]

        map_params = [dst, cond_flags, cond_bounds, conds_op, auto_params]
        self.map_f(map_params)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = EditingWindow()
    sys.exit(app.exec_())
