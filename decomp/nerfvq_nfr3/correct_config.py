import os
import sys

old_root = '/home/zhonghongliang/vqnfr_pro'
new_root = sys.argv[1]

output_root = os.path.join(new_root, 'nerfvq_nfr3/output/train')

def change(path):

    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:

        if old_root in line:
            new_line = line.replace(old_root, new_root)
        else:
            new_line = line

        new_lines.append(new_line)

    with open(path, 'w') as f:
        f.writelines(new_lines)

out_dirs = os.listdir(output_root)
for out_dir in out_dirs:
    cfg_path = os.path.join(output_root, out_dir, 'lr5e-4.ini')
    print(cfg_path)
    change(cfg_path)



