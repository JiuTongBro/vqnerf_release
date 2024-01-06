import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from nerfactor.util import math as mathutil

# https://zhuanlan.zhihu.com/p/160804623
# http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
# https://blog.csdn.net/coldkaweh/article/details/70187399
def get_brdf(pts2l, pts2c, normal, albedo=None, rough=None, f0=None):
    if albedo is None:
        albedo = tf.ones((tf.shape(pts2c)[0], 3), dtype=tf.float32) # Nx3
    if f0 is None:
        f0 = 0.91 * tf.ones((tf.shape(pts2c)[0], 3), dtype=tf.float32) # Nx3
    if rough is None:
        rough = tf.ones((tf.shape(pts2c)[0], 1), dtype=tf.float32) # Nx1

    # Normalize directions and normals
    pts2l = mathutil.safe_l2_normalize(pts2l, axis=2) # NxLx3
    pts2c = mathutil.safe_l2_normalize(pts2c, axis=1) # Nx3
    normal = mathutil.safe_l2_normalize(normal, axis=1) # Nx3
    # Glossy
    h = pts2l + pts2c[:, None, :]  # NxLx3
    h = mathutil.safe_l2_normalize(h, axis=2)
    f = _get_f(pts2c, h, f0)  # NxLx3
    alpha = rough ** 2
    d = _get_d(h, normal, alpha=alpha)  # NxLx1
    g = _get_g(pts2c, pts2l, normal, alpha=alpha)  # NxLx1
    l_dot_n = tf.einsum('ijk,ik->ij', pts2l, normal)[:, :, None]
    v_dot_n = tf.einsum('ij,ij->i', pts2c, normal)[:, None, None]
    denom = 4 * tf.abs(l_dot_n) * tf.abs(v_dot_n)
    brdf_glossy = tf.math.divide_no_nan(f * g * d, denom)  # NxLx3
    # brdf_glossy = tf.tile(microfacet[:, :, None], (1, 1, 3)) # NxLx3
    # Diffuse
    lambert = albedo / np.pi  # Nx3
    brdf_diffuse = tf.broadcast_to(
        lambert[:, None, :], tf.shape(brdf_glossy))  # NxLx3
    # Mix two shaders
    brdf = brdf_glossy + brdf_diffuse  # TODO: energy conservation?
    return brdf, brdf_glossy, brdf_diffuse   # NxLx3

def _get_g(v, l, n, alpha=0.1):
    """GGX.
    """
    alpha = alpha[:, None, :]  # Nx1x1
    g_l = _get_gl(l, n, alpha)
    g_v = _get_gv(v, n, alpha)
    return g_l * g_v  # (n_pts, n_lights, 1)

def _get_gl(l, n, alpha=0.1):
    """GGX.
    """
    cos_theta = tf.einsum('ijk,ik->ij', l, n)[:, :, None] # NxLx1
    cos_theta = tfp.math.clip_by_value_preserve_gradient(cos_theta, 0., 1.) # NxLx1
    cos_theta_sq = tf.square(cos_theta)  # NxLx1
    denom_a = tf.abs(alpha ** 2 + (1 - alpha ** 2) * cos_theta_sq)
    denom = cos_theta + tf.sqrt(denom_a)
    gsub = tf.math.divide_no_nan(2 * cos_theta, denom)
    return gsub #Nxl/1x1

def _get_gv(v, n, alpha=0.1):
    """Schlick-GGX.
    """
    cos_theta = tf.einsum('ij,ij->i', n, v)[:, None, None] # Nx1x1
    cos_theta = tfp.math.clip_by_value_preserve_gradient(cos_theta, 0., 1.) # Nx1x1
    cos_theta_sq = tf.square(cos_theta)  # Nx1x1
    denom_a = tf.abs(alpha ** 2 + (1 - alpha ** 2) * cos_theta_sq)
    denom = cos_theta + tf.sqrt(denom_a)
    gsub = tf.math.divide_no_nan(2 * cos_theta, denom)
    return gsub #Nxlx1

def _get_d(m, n, alpha=0.1):
    """GGX (Trowbridge-Reitz)
    """
    alpha = alpha[:, None, :] #Nx1x1
    cos_theta_m = tf.einsum('ijk,ik->ij', m, n)  #[N,L]
    cos_theta_m = tfp.math.clip_by_value_preserve_gradient(cos_theta_m, 0., 1.)
    cos_theta_m_sq = tf.square(cos_theta_m) #[N,L]
    denom = np.pi * tf.square((cos_theta_m_sq[:, :, None] * (alpha ** 2 -1) + 1)) #[N,L,1]
    d = tf.math.divide_no_nan(alpha ** 2, denom) #[N,L,1]
    return d # (n_pts, n_lights, 1)

def _get_f(v, m, f0=None):
    """Fresnel (Schlick's approximation).
    """
    f0 = f0[:, None, :] # Nx3 -> Nx1x3
    cos_theta = tf.einsum('ijk,ik->ij', m, v)[:, :, None] # NxLx1
    cos_theta = tfp.math.clip_by_value_preserve_gradient(cos_theta, 0., 1.)
    f = f0 + (1 - f0) * (1 - cos_theta) ** 5
    return f  # (n_pts, n_lights, 3)



