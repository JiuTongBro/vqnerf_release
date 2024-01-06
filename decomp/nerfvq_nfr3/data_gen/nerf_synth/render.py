#!/usr/bin/env python

import sys
from os.path import join, basename, exists
import numpy as np
from absl import app, flags
from tqdm import tqdm

from third_party.xiuminglib import xiuminglib as xm
from data_gen.util import read_light, listify_matrix
from nerfactor.util import img as imgutil

import bpy
from mathutils import Matrix


flags.DEFINE_string('scene_path', '', "path to the Blender scene")
flags.DEFINE_string('light_path', '', "path to the light probe")
flags.DEFINE_string('cam_dir', '', "directory containing the camera JSONs")
flags.DEFINE_string(
    'test_light_dir', '', "directory containing the test (novel) light probes")
flags.DEFINE_integer('vali_first_n', 8, "")
flags.DEFINE_float('light_inten', 3, "global scale for the light probe")
flags.DEFINE_integer('res', 512, "resolution of the squre renders")
flags.DEFINE_integer('spp', 128, "samples per pixel")
flags.DEFINE_boolean(
    'add_glossy_albedo', False,
    "whether to add Blender's 'glossy color' to albedo")
flags.DEFINE_string('outdir', '', "output directory")
flags.DEFINE_boolean('overwrite', False, "")
flags.DEFINE_boolean('debug', False, "")
FLAGS = flags.FLAGS


def main(_):
    xm.os.makedirs(FLAGS.outdir, rm_if_exists=FLAGS.overwrite)

    # ------ Render all views

    # For training, validation, testing modes
    for cams_json in xm.os.sortglob(FLAGS.cam_dir, '*', ext='json'):
        mode = basename(cams_json)[:-len('.json')].split('_')[-1]
        print("Mode: " + mode)

        # Load JSON
        data = xm.io.json.load(cams_json)
        cam_angle_x = data['camera_angle_x']
        frames = data['frames']

        if mode == 'val' and FLAGS.vali_first_n is not None:
            frames = frames[:FLAGS.vali_first_n]

        if FLAGS.debug:
            frames = frames[:1]

        # Correct the paths in JSON, to be JaxNeRF-compatible
        data = {'camera_angle_x': cam_angle_x, 'frames': []}
        for i, frame in enumerate(frames):
            folder = f'{mode}_{i:03d}'
            frame['file_path'] = './%s/rgba' % folder
            data['frames'].append(frame)

        json_path = join(FLAGS.outdir, 'transforms_%s.json' % mode)
        xm.io.json.write(data, json_path)

        # Render each frame
        for i, frame in enumerate(tqdm(frames, desc=f"Views ({mode})")):
            cam_transform_mat = frame['transform_matrix']
            folder = f'{mode}_{i:03d}'
            outdir = join(FLAGS.outdir, folder)
            render_view(cam_transform_mat, cam_angle_x, outdir)


def render_view(cam_transform_mat, cam_angle_x, outdir):
    xm.os.makedirs(outdir, rm_if_exists=FLAGS.overwrite)

    # Dump metadata
    metadata_json = join(outdir, 'metadata.json')
    if not exists(metadata_json):
        cam_transform_mat_str = ','.join(
            str(x) for x in listify_matrix(cam_transform_mat))
        data = {
            'scene': basename(FLAGS.scene_path),
            'cam_transform_mat': cam_transform_mat_str,
            'cam_angle_x': cam_angle_x, 'envmap': basename(FLAGS.light_path),
            'envmap_inten': FLAGS.light_inten, 'imh': FLAGS.res,
            'imw': FLAGS.res, 'spp': FLAGS.spp}
        xm.io.json.write(data, metadata_json)

    # Open scene
    xm.blender.scene.open_blend(FLAGS.scene_path)

    # Remove empty tracker that may mess up the camera pose
    objs = [
        x for x in bpy.data.objects if x.type == 'EMPTY' and 'Empty' in x.name]
    bpy.ops.object.delete({'selected_objects': objs})

    # Remove undesired objects
    objs = []
    for o in bpy.data.objects:
        if o.name == 'BackgroundPlane':
            objs.append(o)
    bpy.ops.object.delete({'selected_objects': objs})

    # Remove existing lights, if any
    objs = []
    for o in bpy.data.objects:
        if o.type == 'LIGHT':
            objs.append(o)
        elif o.active_material is not None:
            for node in o.active_material.node_tree.nodes:
                if node.type == 'EMISSION':
                    objs.append(o)
    bpy.ops.object.delete({'selected_objects': objs})

    # Set camera
    cam_obj = bpy.data.objects['Camera']
    cam_obj.data.sensor_width = FLAGS.res
    cam_obj.data.sensor_height = FLAGS.res
    # 1 pixel is 1 mm
    cam_obj.data.lens = .5 * FLAGS.res / np.tan(.5 * cam_angle_x)
    # NOTE: If not wrapping the NumPy array as a Matrix, it would be transposed
    # for some unknown reason: https://blender.stackexchange.com/q/159824/30822
    cam_obj.matrix_world = Matrix(cam_transform_mat)
    bpy.context.view_layer.update()

    # Add environment lighting
    xm.blender.light.add_light_env(
        env=FLAGS.light_path, strength=FLAGS.light_inten)

    # Rendering settings
    xm.blender.render.easyset(w=FLAGS.res, h=FLAGS.res, n_samples=FLAGS.spp)
    # if args.direct_only:
    #     bpy.context.scene.cycles.max_bounces = 0

    # Render RGBA
    rgba_png = join(outdir, 'rgba.png')
    if not exists(rgba_png):
        xm.blender.render.render(rgba_png, cam=cam_obj)
    rgba = xm.io.img.read(rgba_png)
    alpha = rgba[:, :, 3]
    alpha = xm.img.normalize_uint(alpha)

    def render_composition(mode='diffuse'):
        color_png = join(outdir, mode + '_rgb.png')
        if not exists(color_png):
            if not (mode=='emit' or mode=='environment'):
                direct_exr = join(outdir, mode + '-direct.exr')
                xm.blender.render.render_lighting_passes(
                    direct_exr, cam=cam_obj, select=mode + '_direct')
                direct = xm.io.exr.read(direct_exr)

                indirect_exr = join(outdir, mode + '-indirect.exr')
                xm.blender.render.render_lighting_passes(
                    indirect_exr, cam=cam_obj, select= mode + '_indirect')
                indirect = xm.io.exr.read(indirect_exr)

                color_exr = join(outdir, mode + '-color.exr')
                xm.blender.render.render_lighting_passes(
                    color_exr, cam=cam_obj, select= mode + '_color')
                color = xm.io.exr.read(color_exr)

                final_color = (direct + indirect) * color
                final_color = np.dstack((final_color, alpha))
                xm.io.img.write_arr(final_color, color_png, clip=True)
            else:
                color_exr = join(outdir, mode + '.exr')
                xm.blender.render.render_lighting_passes(
                    color_exr, cam=cam_obj, select=mode)
                color = xm.io.exr.read(color_exr)

                final_color = np.dstack((color, alpha))
                xm.io.img.write_arr(final_color, color_png, clip=True)

    render_composition('diffuse')
    render_composition('glossy')
    render_composition('transmission')
    render_composition('emit')
    render_composition('environment')


    # Render albedo
    # Let's assume white specularity, so the diffuse_color alone is albedo
    albedo_png = join(outdir, 'albedo.png')
    if not exists(albedo_png):
        diffuse_color_exr = join(outdir, 'diffuse-color.exr')
        diffuse_color = xm.io.exr.read(diffuse_color_exr)
        if FLAGS.add_glossy_albedo:
            glossy_color_exr = join(outdir, 'glossy-color.exr')
            glossy_color = xm.io.exr.read(glossy_color_exr)
        else:
            glossy_color = np.zeros_like(diffuse_color)
        albedo = diffuse_color + glossy_color
        albedo = np.dstack((albedo, alpha))
        xm.io.img.write_arr(albedo, albedo_png, clip=True)

    # Render normals ...
    normal_png = join(outdir, 'normal.png')
    if not exists(normal_png):
        normal_exr = join(outdir, 'normal.exr')
        normal_refball_exr = join(outdir, 'refball-normal.exr')
        xm.blender.render.render_normal(
            normal_exr, cam=cam_obj, world_coords=True,
            outpath_refball=normal_refball_exr)
        normals = xm.io.exr.read(normal_exr)
        xm.vis.geometry.normal_as_image(
            normals, alpha, outpath=normal_png, keep_alpha=True)
        # and also normals of the reference ball
        normals_refball = xm.io.exr.read(normal_refball_exr)
        normal_refball_png = normal_refball_exr[:-len('.exr')] + '.png'
        xm.vis.geometry.normal_as_image(
            normals_refball, outpath=normal_refball_png, keep_alpha=True)

    # Render relit ground truth
    if FLAGS.test_light_dir is not None:
        # With HDR maps
        for envmap_path in xm.os.sortglob(FLAGS.test_light_dir, '*.hdr'):
            envmap_name = basename(envmap_path).split('.')[0]
            outpath = join(outdir, 'rgba_%s.png' % envmap_name)
            if exists(outpath):
                continue
            xm.blender.light.add_light_env(env=envmap_path, strength=1.)
            xm.blender.render.render(outpath, cam=cam_obj)
        # With OLAT
        for envmap_path in xm.os.sortglob(FLAGS.test_light_dir, '*.json'):
            envmap_name = basename(envmap_path).split('.')[0]
            outpath = join(outdir, 'rgba_%s.png' % envmap_name)
            if exists(outpath):
                continue
            olat = xm.io.json.load(envmap_path)
            # NOTE: not using intensity in JSON; because Blender uses Watts
            # (and fall-off), it's impossible to match exactly our predictions
            xm.blender.light.add_light_env(
                env=(1, 1, 1, 1), strength=0)  # ambient
            pt_light = xm.blender.light.add_light_point(  # point
                xyz=olat['point_location'], energy=50_000)
            xm.blender.render.render(outpath, cam=cam_obj)
            xm.blender.object.remove_objects(pt_light.name)  # avoid light accu.







if __name__ == '__main__':
    # Blender-Python binary
    argv = sys.argv
    if '--' in argv:
        arg_i = argv.index('--')
        argv = argv[(arg_i - 1):arg_i] + argv[(arg_i + 1):]

    app.run(main=main, argv=argv)