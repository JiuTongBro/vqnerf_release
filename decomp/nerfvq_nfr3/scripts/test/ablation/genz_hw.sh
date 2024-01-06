scene="$1"
gpus="$2"
model='nfr_unit'
overwrite='False'
proj_root='/home/zhonghongliang/vqnfr_pro'
repo_dir="$proj_root/nerfvq_nfr3"
viewer_prefix='' # or just use ''

data_type='hw'

# I. Shape Pre-Training
data_root="$proj_root/data/1115_hw_data/1115data_1/$scene"
imh='420'
if [[ "$scene" == redcar_-1 ]]; then
    light_init_val='0.5'
else
    light_init_val='0.7'
fi

use_nerf_alpha='False'

surf_root="$repo_dir/output/hw_surf/${scene}"
shape_outdir="$repo_dir/output/train/${scene}_shape"
light_path="$proj_root/data/train_envs/3072.hdr"
unit_root="$proj_root/my_data/comps/${scene}"

test_envmap_dir="$proj_root/data/test_envs/hw"
outroot="$repo_dir/output/train/${scene}_${model}"

lr='5e-4'
# III. Simultaneous Relighting and View Synthesis (testing)
ckpt="$outroot/lr${lr}/checkpoints/ckpt-10"
function="genz"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/gen_z.sh" "$gpus" --ckpt="$ckpt" --function="$function"