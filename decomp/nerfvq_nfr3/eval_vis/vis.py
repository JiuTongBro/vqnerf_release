import json
import os
import numpy as np
import pandas as pd

root = 'full_metric'
vis_root = 'vis'
models = ['ref_nfr']
f_jsons = {}

for model in models:
    with open(os.path.join(root, model + '.json')) as f:
        f_jsons[model] = json.load(f)

for scene in f_jsons[models[0]].keys():
    print('\r#- ', scene)

    model_dict = {}
    for m in models: model_dict[m] = []

    col_indexs = []
    # ['rgb', 'diff', 'spec', 'env',  'kd', 'ks', 'rough']
    for mode in ['rgb', 'kd', 'ks', 'rough']:
        print('- ', mode)
        for metric in ['psnr', 'ssim', 'lpips']:
            col_indexs.append(mode+'_'+metric)

            for m in models:
                if not mode in f_jsons[m][scene].keys():
                    print('Skip ', mode, ' in ', m, ' ', scene)
                    model_dict[m].append(0.)
                else: model_dict[m].append(np.mean(f_jsons[m][scene][mode][metric]))

    df = pd.DataFrame(model_dict, index=col_indexs)
    df.to_csv(os.path.join(vis_root, scene+'.csv'))




