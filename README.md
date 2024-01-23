# VQ-NeRF

This is the code release for [VQ-NeRF Home Page](https://jtbzhl.github.io/VQ-NeRF.github.io/).


## Acknowledgements

Our code is mainly built upon the following projects. We sincerely thank the authors:
- [NeRFactor](https://github.com/google/nerfactor)
- [NeuS](https://github.com/Totoro97/NeuS)
- [Sonnet](https://github.com/google-deepmind/sonnet)

Also, we would like to thank all the collaborators who have helped with this project.


## Overview

Clone this repository and rename it:

```shell
git clone https://github.com/JiuTongBro/vqnerf_release.git
mv vqnerf_release vqnfr_pro_release
cd vqnfr_pro_release
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

The `data/test_envs` stores a total of 16 envrionment maps for relighting. Eight of them are released by the [nvdiffrec](https://github.com/NVlabs/nvdiffrec), and the other eight are collected by us. For some types of the datasets, we flipped those illumination, as the 'upper' direction is reversed in their coordinates.


## Pretrained Weights

The pretrained weights can be found [here](https://drive.google.com/drive/folders/1CZcpFUSitfyiVPQvkluibs-mf-laxJUE?usp=sharing). The `exp/` folder contains the weights for geometry reconstruction, and the `output/` folder contains the weights for decomposition and segmentation.


## Geometry Reconstruction

Go to the `geo/` folder, prepare and activate the environment.

```shell
cd geo
conda create --prefix="./geo_env" python=3.6
conda activate ./geo_env
pip install -r NeuS-ours2/requirements.txt
# You may need to manually update the torch-1.8.0 according to your cuda version
pip install tensorboard
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
pip install opencv-python==4.5.4.60
```

- In some cases, the environment installation of the `decomp/` may lead to an issue relevant to CUDA 12:
```
Could not load library libcublasLt.so.12. Error: libcublasLt.so.12: cannot open shared object file: No such file or directory
Aborted (core dumped)
```
This suggests that some libs are missed in your CUDA. To avoid mess up your CUDA env, you can manually download and place them, following [this link](https://stackoverflow.com/questions/76646474/could-not-load-library-libcublaslt-so-12-error-libcublaslt-so-12-cannot-open) for solution.

- We found that multiple factors of the running environment (e.g. tf-cuda-cudnn versions) could affect the floating point error in tf-gpu. And the accumulated error could slightly influence the reproducibility of the experiments. To reproduce our results stably, you may download our [pretrained weights](https://drive.google.com/drive/folders/1CZcpFUSitfyiVPQvkluibs-mf-laxJUE?usp=sharing).

(Optional) Check the environment:

```shell
cd nerfvq_nfr3
python check_env.py # Optional, check the environments
cd ../
```

Then, link the data and the extracted geometry:

```shell
ln -s <project_root>/data ./
ln -s <project_root>/geo/NeuS-ours2/surf/* ./nerfvq_nfr3/output/
cd nerfvq_nfr3
```

Then follow the instructions in [decomp/nerfvq_nfr3](https://github.com/JiuTongBro/vqnerf_release/tree/main/decomp/nerfvq_nfr3).


## FAQ

- For the convenience of releasement, the codes, weights, and dependencies have been re-organized. Shall there be any problem, please open an issue. We will try to assist if we find time.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
