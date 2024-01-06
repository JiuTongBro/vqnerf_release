scene="$1"
gpus="$2"
proj_root="/home/zhonghongliang/vqnfr_pro_release/decomp"
repo_dir="$proj_root/nerfvq_nfr3/scripts/train"

if [[ "$scene" == drums_3072 || "$scene" == lego_3072 || "$scene" == hotdog_2163 || "$scene" == ficus_2188 || "$scene" == materials_2163]]; then
        "${repo_dir}/nfr_nerf.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/vq_nerf.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/ref_nerf.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == chair0_3072 || "$scene" == kitchen6_7095 || "$scene" == machine1_3072 ]]; then
        "${repo_dir}/nfr_mat.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/vq_mat.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/ref_mat.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == rabbit_-1 || "$scene" == hwchair_-1 || "$scene" == toyrabbit_-1 || "$scene" == redcar_-1 ]]; then
        "${repo_dir}/nfr_hw.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/vq_hw.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/ref_hw.sh" "$scene" "$gpus" "$proj_root"
elif [[ "$scene" == dtu_scan24 || "$scene" == dtu_scan69 || "$scene" == dtu_scan110 ]]; then
        "${repo_dir}/nfr_dtu.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/vq_dtu.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/ref_dtu.sh" "$scene" "$gpus" "$proj_root"
else
        "${repo_dir}/nfr_ours.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/vq_ours.sh" "$scene" "$gpus" "$proj_root"
        "${repo_dir}/ref_ours.sh" "$scene" "$gpus" "$proj_root"
fi