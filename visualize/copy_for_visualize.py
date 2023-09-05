root="/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/results/comparison_motivation"
expname_list=["train11_on_thuman2_36views_pamir_wopesdf_{2}_run111", "train11_on_THUMAN2_36views_pamir_only_debug1", "train11_on_thuman2_36views_pamir_pesdf_{16}_run111", "train11_on_thuman2_36views_pamir_pesdf_{14}_run111", "train11_on_thuman2_36views_pamir_pesdf_{12}_run111", "train11_on_thuman2_36views_pamir_pesdf_{10}_run111"]
# name="00159-shortlong-pose_model-000150"
# name="00134-longlong-frisbee_trial1-000190"
# name="00134-longlong-stretch_trial1-000450"
# name="03375-shortshort-climb_trial1-000170"
name="03375-shortshort-hands_up_trial1-000270"
rotation="120"
dataset="cape" ## or thuman2
intent_of_visulization="motivation_pamir_wopesdf_pamir_pesdf"
target_dir=f"/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/results/comparison_motivation/visualize/{intent_of_visulization}_{name}"
import os
os.makedirs(target_dir, exist_ok=True)
from os.path import join as opj
import shutil
for expname in expname_list:
  final_path=opj(root, expname, dataset, name)
  # print(final_path)
  source_imgpath=opj(final_path, f"{rotation}_nc.png")
  print(source_imgpath)
  target_imgpath=opj(target_dir, f"{expname}_{rotation}_nc.png")
  print(target_imgpath)
  shutil.copy(source_imgpath, target_imgpath)
# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/results/comparison_motivation/train11_on_thuman2_36views_pamir_wopesdf_{2}_run111/cape/00159-shortlong-pose_model-000150

# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/results/comparison_motivation/train11_on_thuman2_36views_pamir_wopesdf_{2}_run111/cape/00159-shortlong-pose_model-000150

# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/results/comparison_motivation/train11_on_thuman2_36views_pamir_pesdf_{14}_run111/cape/03375-shortshort-hands_up_trial1-000270/