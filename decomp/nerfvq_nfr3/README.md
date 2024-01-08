# VQ-NeRF: Main Folder

This is the main folder for decomposition, segmentation, relighting, and editing.

The training, testing and editing scripts are `train.sh`, `test.sh` and `edit.sh` under the `scripts/` folder. Before running the scripts, you need to set the `proj_root` to your own `decomp/` path in all the three files.


## Training and Testing

```shell
bash scripts/train.sh <scene_name> <gpu> # training
bash scripts/test.sh <scene_name> <gpu> # testing
# e.g. bash scripts/train.sh drums_3072 0
```

- If you run into problems with the `.sh` permissions (Permission denied), go to the corresponding folder, and run:

```shell
# sed -i 's/\r//' *.sh, this is used to align the Windows .sh format to Linux (Optional, handling '/r')
chmod 777 *.sh
```

After training and testing:
- The reconstruction and decomposition results can be found in `output/train/<scene_name>_ref_nfr/lr<lr>/pd_test`
- The relighting results can be found in `output/train/<scene_name>_ref_nfr/lr<lr>/pd_relit`
- The segmentation results can be found in `output/train/<scene_name>_ref_nfr/lr<lr>/pd_vq`

More explanations can be found in [scripts/](https://github.com/JiuTongBro/vqnerf_release/tree/main/decomp/nerfvq_nfr3/scripts).


## Evaluation

```shell
# Firstly, you may need to configure the paths and flags in the python files
python eval/metric_eval.py <dataset_type> ref_nfr # reconstruction, decomposition and relighting evaluation
python eval/cluster_eval.py # segmentation evaluation, only the nerf dataset has GT.
# e.g. python eval/metric_eval.py nerf ref_nfr
```

You can also use the `eval/vis.py` and `eval/cluster_vis.py` to convert the scores to `.csv` format.


## Editing

To run our editing UI, you need a local computer and a remote server.

First, prepare a local cache folder, download the generated `pd_vq/` folder to local and rename it to `pd_comps/`. Then download the `data/test_envs/vis/` to this local directory too.

Then, run the following command on the server to make the server ready for editing:

```shell
bash scripts/edit.sh <scene_name> <gpu>
```

Afterwards, configure the parameters and paths in the `__init__()` of `ui4.py`, inlcuding:

- `scene`: the edited scene name.
- `self.local_folder`: the path of your local cache folder.
- `self.server_root`: the path of the code root on your server.
- the sever IP, username and pwd in `ssh.connect`.

Now, you can run the `ui4.py` in local for editing. Note that the communication with the server is based on uploading/downloading, so some operations may take a while.



