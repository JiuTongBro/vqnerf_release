# VQ-NeRF

This is the folder for Geometry Reconstruction.


## Training

For `nerf`, `mat` and `hw` scenes, run:

```shell
python nerf_runner.py --case <scene_name>
```

For `dtu` and `ours` scenes, run:

```shell
python dtu_runner.py --case <scene_name>
```

Run the training with the defualt configs may require 48G GPU memory for some scenes. You may reduce the `batch_size` in the config files to avoid OOM. The config files are under `confs/`, you can refer to the `models/helpers.py` for correspondence.

### Generate Geometry

After training, we need to extract the geometry from different views.

For `nerf`, `mat` and `hw` scenes, run:

```shell
python gen_geo.py --case <scene_name>
```

For `dtu` and `ours` scenes, run:

```shell
python dtu_geo.py --case <scene_name>
```

As the geometry occlusion in CG data are more complex, we also introduce the visibility term in the rendering of CG scenes to better model the shadows. The extraction of the visibility terms is slow (take several hours or even 2-3 days). And the extracted data takes up a lot stoage (about 50G for one scene with about 100 views).

You may accelerate it by running the geometry extraction on multiple GPUs, e.g.:

```shell
CUDA_VISIBLE_DEVICES=0 python gen_geo.py --case <scene_name> --num_p 2 --p_i 0 & CUDA_VISIBLE_DEVICES=1 python gen_geo.py --case <scene_name> --num_p 2 --p_i 1
```
This command divides the geomtry extraction into `num_p` parallel tasks, and the `p_i` indicates which subprocesses it is.

The visibility term is only related to the occlussion and shadowing. You may also omit this term, especially in scenes with simple geometry, like in some other methods. This can be a trade off between model accuracy and efficiency:

```shell
python gen_geo.py --case <scene_name> --no_vis
```

The geometry extraction for video rendering follows a similar way.

