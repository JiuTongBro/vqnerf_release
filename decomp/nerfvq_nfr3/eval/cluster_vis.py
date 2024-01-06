import json
import os
import numpy as np
import pandas as pd

vis_root = 'cluster_vis'

# with open(os.path.join('cluster.json')) as f:
with open(os.path.join('ind_abl.json')) as f:
    f_jsons = json.load(f)

scenes = ['drums_3072', 'lego_3072', 'hotdog_2163', 'ficus_2188', 'materials_2163']

for i in range(len(scenes)):
    scene = scenes[i]
    print('\r#- ', scene)

    model_dict = {}
    for m in f_jsons.keys():
        model_dict[m] = []

    col_indexs = []

    for metric in ['purity', 'f1-micro', 'f1-macro', 'p-macro', 'r-macro',]:
        col_indexs.append(metric)

        for m in f_jsons.keys():
            model_dict[m].append(f_jsons[m][metric][i])

    df = pd.DataFrame(model_dict, index=col_indexs)
    df.to_csv(os.path.join(vis_root, scene+'.csv'))




