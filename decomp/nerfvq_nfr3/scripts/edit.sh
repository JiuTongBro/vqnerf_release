scene="$1"
gpus="$2"
proj_root="/home/zhonghongliang/vqnfr_pro_release/decomp"
repo_dir="$proj_root/nerfvq_nfr3/scripts/test"

if [[ "$scene" == drums_3072 || "$scene" == lego_3072 || "$scene" == hotdog_2163 || "$scene" == ficus_2188 || "$scene" == materials_2163]]; then
        "${repo_dir}/edit_nerf.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == chair0_3072 || "$scene" == kitchen6_7095 || "$scene" == machine1_3072 ]]; then
        "${repo_dir}/edit_mat.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == rabbit_-1 || "$scene" == hwchair_-1 || "$scene" == toyrabbit_-1 || "$scene" == redcar_-1 ]]; then
        "${repo_dir}/edit_hw.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == dtu_scan24 || "$scene" == dtu_scan69 || "$scene" == dtu_scan110 ]]; then
        "${repo_dir}/edit_dtu.sh" "$scene" "$gpus" "$proj_root"
else
        "${repo_dir}/edit_ours.sh" "$scene" "$gpus" "$proj_root"
fi