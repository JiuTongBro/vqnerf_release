scene="$1"
gpus="$2"
model='ref_nfr'
overwrite='False'
proj_root="$3"
repo_dir="$proj_root/nerfvq_nfr3"
viewer_prefix='' # or just use ''

data_type='dtu'

# I. Shape Pre-Training
data_root="$proj_root/data/dtu_split2/$scene"
imh='512'
light_init_val='0.7'

use_nerf_alpha='False'

surf_root="$repo_dir/output/dtu_surf/${scene}"
shape_outdir="$repo_dir/output/train/${scene}_shape"
light_path="$proj_root/data/train_envs/3072.hdr"
unit_root="$proj_root/my_data/comps/${scene}"

# VQ

nfr_outdir="$repo_dir/output/train/${scene}_vq_nfr"
nfr_ckpt="$nfr_outdir/lr5e-4/checkpoints/ckpt-5"

if [[ "$scene" == dtu_scan24 ]]; then
    # Real scenes: NeRF & DTU
    no_brdf_chunk='False'
else
    no_brdf_chunk='True'
fi

test_envmap_dir="$proj_root/data/test_envs/dtu"
outroot="$repo_dir/output/train/${scene}_${model}"
REPO_DIR="$repo_dir" "$repo_dir/nerfactor/trainvali_run.sh" "$gpus" --config="$model.ini" --config_override="data_root=$data_root,imh=$imh,use_nerf_alpha=$use_nerf_alpha,data_nerf_root=$surf_root,test_envmap_dir=$test_envmap_dir,outroot=$outroot,viewer_prefix=$viewer_prefix,overwrite=$overwrite,light_path=$light_path,unit_root=$unit_root,light_init_val=$light_init_val,data_type=$data_type,env_inten=$env_inten,olat_inten=$olat_inten,nfr_model_ckpt=$nfr_ckpt,no_brdf_chunk=$no_brdf_chunk"