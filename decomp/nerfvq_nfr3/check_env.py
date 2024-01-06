import tensorflow as tf
import numpy as np
from third_party.xiuminglib import xiuminglib as xm

psnr = xm.metric.PSNR('uint8')
ssim = xm.metric.SSIM('uint8')
lpips = xm.metric.LPIPS('uint8')

a = np.ones((2, 3), dtype=np.uint8)
b = np.ones([72, 72, 3], dtype=np.uint8) * 200
c = np.ones([72, 72, 3], dtype=np.uint8) * 190
print(a.shape)

d = xm.linalg.normalize(a, axis=1)
print(d)

e = psnr(b, c)
print(e)
e = ssim(b, c)
print(e)
e = lpips(b, c)
print(e)

print('# Is tf-gpu avliable: ', tf.test.is_gpu_available())