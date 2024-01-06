import cv2
import numpy as np
import os
import sys


scene = sys.argv[1]
mode = sys.argv[2] # recon, relit, edit
env = sys.argv[3]

add_env = False
mode_map = {'recon': 'vis_video',
            'relit': 'video_relight',
            'edit': 'video_edit',}

vis_env =  '/home/zhonghongliang/vqnfr_pro/data/test_envs/vis/'
fps = 20

print(scene)
ours_root = '/home/zhonghongliang/vqnfr_pro/nerfvq_nfr3/output/train/'+scene+'_ref_nfr/'
ours_dir = ours_root + 'lr5e-4/'+mode_map[mode]+'/'
views = os.listdir(ours_dir)
views.sort()

img_template = cv2.imread(ours_dir + 'test_000/gt_rgb.png')
h,w = img_template.shape[:2]
env_h = h // 10
print(h, w)

dst_root = ours_root + scene + '_' + env + '.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(dst_root, fourcc, fps, (w, h))

for view in views:
    if not view[:4] == 'test': continue

    if env == 'rgb': img_f = ours_dir + view + '/pred_rgb.png'
    else: img_f = ours_dir + view + '/pred_rgb_probes_' + env + '.png'
    pred_img = cv2.imread(img_f)
    if pred_img is None:
        print('img error: ', img_f)
        break

    if add_env and (not env == 'rgb'):
        f_env = vis_env + env + '.png'
        env_img = cv2.imread(f_env)
        env_img = cv2.resize(env_img, (env_h * 2, env_h))
        pred_img[:env_h, :env_h * 2] = env_img
    videoWriter.write(pred_img)

videoWriter.release()





