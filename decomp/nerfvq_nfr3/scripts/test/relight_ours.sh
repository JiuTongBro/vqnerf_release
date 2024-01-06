scene="$1"
gpus="$2"
model='ref_nfr'
overwrite='False'
proj_root="$3"
repo_dir="$proj_root/nerfvq_nfr3"
viewer_prefix='' # or just use ''

data_type='dtu'

# I. Shape Pre-Training
data_root="$proj_root/data/colmap_split/$scene"
imh='420'
light_init_val='1.'

use_nerf_alpha='False'

surf_root="$repo_dir/output/ours_surf/${scene}"
shape_outdir="$repo_dir/output/train/${scene}_shape"
light_path="$proj_root/data/train_envs/3072.hdr"
unit_root="$proj_root/my_data/comps/${scene}"

# VQ

nfr_outdir="$repo_dir/output/train/${scene}_vq_nfr"
nfr_ckpt="$nfr_outdir/lr5e-4/checkpoints/ckpt-5"

test_envmap_dir="$proj_root/data/test_envs/dtu"
outroot="$repo_dir/output/train/${scene}_${model}"

lr='5e-4'
# III. Simultaneous Relighting and View Synthesis (testing)
ckpt="$outroot/lr${lr}/checkpoints/ckpt-5"
function="relight"

REPO_DIR="$repo_dir" "$repo_dir/nerfactor/test_run.sh" "$gpus" --ckpt="$ckpt" --function="$function"