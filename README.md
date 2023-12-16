# VQ-NeRF

This is the code release for [VQ-NeRF Home Page](https://jtbzhl.github.io/VQ-NeRF.github.io/).


## Acknowledgements

Our codes is mainly built upon the following projects. We sincerely thank the authors:
- [NeRFactor](https://github.com/google/nerfactor)
- [NeuS](https://github.com/Totoro97/NeuS)
- [Sonnet](https://github.com/google-deepmind/sonnet)


## Overview

Clone this repository:

```shell
https://github.com/JiuTongBro/vqnerf_release.git
```

Our method requires a two-stage running. The first stage is for the geometry reconstruction, and the second stage is for the decomposition and segmentation.
- The codes for geometry reconstruction are under the `geo/` folder. It is an edited version of [NeuS](https://github.com/Totoro97/NeuS)
- The codes for decomposition and segmentation are under the `decomp/` folder. It is modified based on [NeRFactor](https://github.com/google/nerfactor). This is the main part of our codes.


## Data

Download the data from [this link](https://drive.google.com/drive/folders/1YjWhKcip-nEvheOjzb1epgNSkJi8K9UL?usp=sharing). Then place all the items under the `data/` folder.

There are five types of datasets:
- `nerf` dataset: It is a CG dataset, containing five NeRF-Blender scenes. The GT images are stored in `data/nfr_blender`, the GT albedos and relighted images are stored in `data/vis_comps`, the GT segmentation labels are stored in `data/nerf_seg1`.
- `mat` dataset: It is a CG dataset collected by us, containing three scenes with GTs for all BRDF attributes. It is stored in `data/mat_blender`.
- `dtu` dataset: It is a real dataset, containing three scenes collected from the NeuS-DTU dataset. It is stored in `data/dtu_split2`.
- `ours` dataset: It is a real dataset collected by us, containing three scenes. It is stored in `data/colmap_split`.
- `hw` dataset: It is a real dataset collected by us, containing four scenes. It is stored in `data/1115_hw_data/1115data_1`.

The coordinate system for `nerf`, `mat` and `hw` dataset follows NeRF-Blender, while the coordinate system for `dtu` and `ours` dataset follows NeuS-DTU.

The `data/test_envs` stores a total of 16 envrionment maps for relighting. Eight of them are released by the [nvdiffrec](https://github.com/NVlabs/nvdiffrec), and the other eight are collected by us. For some types of the datasets, we flipped those illumination, as the 'upper' direction is reversed in those scenes.


## Geometry Reconstruction

Go to the `geo/` folder, prepare and activate the environment.

```shell
cd geo
conda create --prefix="./geo_env" python=3.6
conda activate ./geo_env
pip install -r NeuS-ours2/requirements.txt
```

Go to the code folder and link the data from the project root:

```shell
cd NeuS-ours2
ln -s <project_root>/data ./
```

Then follow the instructions in [geo/NeuS-ours2](https://github.com/JiuTongBro/vqnerf_release/tree/main/geo/NeuS-ours2).

## Decomposition and Segmentation

Go to the `decomp/` folder, prepare and activate the environment.

```shell
cd decomp
conda create --prefix="./decomp_env" python=3.6
conda activate ./decomp_env
pip install -r nerfvq_nfr3/requirements.txt # may need some manual adjustments, like the torch-cuda correspondence
python check_env.py # Optional, check the environments
```

(Optional) In some cases, there may be an issue relevant to CUDA 12:
```
Could not load library libcublasLt.so.12. Error: libcublasLt.so.12: cannot open shared object file: No such file or directory
Aborted (core dumped)
```
This suggests that some libs are missed in your CUDA. To avoid mess up your CUDA env, you can manually download and place them, following [this link](https://stackoverflow.com/questions/76646474/could-not-load-library-libcublaslt-so-12-error-libcublaslt-so-12-cannot-open) for solution.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
