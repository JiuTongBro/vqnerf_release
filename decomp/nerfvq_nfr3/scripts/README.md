# VQ-NeRF: Main Folder - Scripts

## Training

The training can be divided into three stages.

- `nfr_<data_type>.sh`: Warm up. VQ convergence is heavily affected by the codebook initialization. So we first warm up the continuous branch, cluster the latent vectors with kmeans, and initialize the VQ codebook with the cluster centers.
- `vq_<data_type>.sh`: Decomposition and Segmentation. This is the main part of our method, the decomposition and segmentation. Note that the VQ codebook is updated by EMA, but not gradient propagation. This may cause warnings, and it can be ignored.
- `ref_<data_type>.sh`: Residual Baking. Compared to the radiance fields which can more flexibly model the scene color, BRDF decomposition may lead to detail losses in the reconstructed appearance (e.g. loss of intra-scene reflections due to the lack of suitable modeling). To efficiently add these details back, we implicitly bake an appearance residual into the reconstructed RGBs through a reference RGB generated in the geometry reconstruction (they do not undergo decomposition and therefore retain more details). This step will only update the reconstructed RGB results in testing.

## Testing

The testing/inference can be divided in to four stages.

- Raw reconstruction and decomposition. Generate the raw results of the RGB reconstruction and the BRDF decomposition. The outputs are stored in `raw_test/`.
- Scaled decomposition. Generate the BRDF-scaled results of the decomposition. The outputs are stored in `pd_test/`.
- Scaled relighting. Generate the BRDF-scaled results of the relighting. The outputs are stored in `pd_relit/`.
- Segmentation. Generate the results of the segmentation. The outputs are stored in `pd_vq/`.



