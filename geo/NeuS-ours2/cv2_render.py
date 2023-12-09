import cv2
import numpy as np
import os

mode = 'test'
scenes = ['drums_3072', 'materials_2163']
fps = 20

for scene in scenes:
    print(scene)
    ours_root = '/home/zhonghongliang/neus_pro/NeuS-ours2/video/'
    ours_dir = ours_root+scene+'/'
    views = os.listdir(ours_dir)
    views.sort()
    print(views)

    img_template = cv2.imread(ours_dir + 'test_000/rgb.png')
    h,w = img_template.shape[:2]
    print(h, w)

    dst_root = ours_root + scene
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(os.path.join(ours_root, scene + '.avi'), fourcc, fps, (w, h))

    for view in views:
        if not view[:4] == 'test': continue

        img_f = ours_dir + view + '/rgb.png'
        pred_img = cv2.imread(img_f, -1)

        mask_f = ours_dir + view + '/alpha.png'
        mask_img = cv2.imread(mask_f, -1)[..., None]
        alpha_mask = np.where(mask_img > 0, 1, 0).astype(np.uint8)
        pred_img = pred_img * alpha_mask + (255 - mask_img)

        if pred_img is None:
            print('img error!')
            continue

        videoWriter.write(pred_img)
    videoWriter.release()



