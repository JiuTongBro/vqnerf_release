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
            break


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

    # Render roughness
    mat_exr = join(outdir, 'mat_ind.exr')
    xm.blender.render.render_rough(
        mat_exr, cam=cam_obj, select='material_index')



if __name__ == '__main__':
    # Blender-Python binary
    argv = sys.argv
    if '--' in argv:
        arg_i = argv.index('--')
        argv = argv[(arg_i - 1):arg_i] + argv[(arg_i + 1):]

    app.run(main=main, argv=argv)