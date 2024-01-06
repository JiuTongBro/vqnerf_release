scene="$1"
gpus="$2"
model='nfr_unit'
overwrite='False'
proj_root="$3"
repo_dir="$proj_root/nerfvq_nfr3"
viewer_prefix='' # or just use ''

data_type='hw'

# I. Shape Pre-Training
data_root="$proj_root/data/1115_hw_data/1115data_1/$scene"
imh='420'
light_init_val='0.7'

use_nerf_alpha='False'

surf_root="$repo_dir/output/hw_surf/${scene}"
shape_outdir="$repo_dir/output/train/${scene}_shape"
light_path="$proj_root/data/train_envs/3072.hdr"
unit_root="$proj_root/my_data/comps/${scene}"

test_envmap_dir="$proj_root/data/test_envs/hw"
outroot="$repo_dir/output/train/${scene}_${model}"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,test_envmap_dir=$test_envmap_dir,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite,light_path=$light_path,unit_root=$unit_root,light_init_val=$light_init_val,data_type=$data_type"