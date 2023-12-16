# VQ-NeRF: Geometry Reconstruction

This is the folder for Geometry Reconstruction.


## Training

For `nerf`, `mat` and `hw` scenes, run:

```shell
python nerf_runner.py --case <scene_name>
# e.g. python nerf_runner.py --case drums_3072
```

For `dtu` and `ours` scenes, run:

```shell
python dtu_runner.py --case <scene_name>
```

Run the training with the defualt configs may require 48G GPU memory for some scenes. You may reduce the `batch_size` in the config files to avoid OOM. The config files are under `confs/`, you can refer to the `models/helpers.py` for correspondence.

## Generate Geometry

After training, we need to extract the geometry from different views.

For `nerf`, `mat` and `hw` scenes, run:

```shell
python gen_geo.py --case <scene_name>
```

For `dtu` and `ours` scenes, run:

```shell
python dtu_geo.py --case <scene_name>
```

As the geometry occlusion in CG data are more complex, we also include the visibility term in the rendering of CG scenes to better model the shadows. Pleae reserve enough space to store the extracted data (about 50G for each CG scene).

- (Optional) Accelerations:

The extraction of the visibility term may be extremely slow on some scenes. You may accelerate it by running the geometry extraction on multiple GPUs, e.g.:

```shell
CUDA_VISIBLE_DEVICES=0 python gen_geo.py --case <scene_name> --num_p 2 --p_i 0 & CUDA_VISIBLE_DEVICES=1 python gen_geo.py --case <scene_name> --num_p 2 --p_i 1
```
This command divides the geomtry extraction into `num_p` parallel tasks, and the `p_i` indicates which subprocesses it is.

