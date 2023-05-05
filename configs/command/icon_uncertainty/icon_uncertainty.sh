

source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh
# conda activate /mnt/cephfs/home/qiuzhen/anaconda3/envs/icon3090v1/ 
conda activate icon3090v1
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 

# CUDA_VISIBLE_DEVICES=6 python -m apps.train -cfg configs/train/icon_uncertainty/icon-filter_uncertainty_beta0003.yaml --gpus 0 --num_gpus 1  --uncertainty --beta_min 0.003 --beta_plus 5 --test_mode

# CUDA_VISIBLE_DEVICES=6,7 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty_beta0003.yaml --gpus 0 --num_gpus 2  --uncertainty --beta_min 0.003 --beta_plus 5 --test_mode
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 1 2  --occ 0.8


# CUDA_VISIBLE_DEVICES=3,5 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty.yaml --gpus 0 --num_gpus 2 --name baseline/pertubesdf0005 --perturb_sdf 0.005
# CUDA_VISIBLE_DEVICES=3 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty.yaml --gpus 0 --num_gpus 1 --name baseline/pertubesdf0001 --perturb_sdf 0.001

# CUDA_VISIBLE_DEVICES=2 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty.yaml --gpus 0 --num_gpus 1 --name baseline/pertubesdf01 --perturb_sdf 0.1
# CUDA_VISIBLE_DEVICES=3 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty.yaml --gpus 0 --num_gpus 1 --name baseline/pertubesdf02 --perturb_sdf 0.2
CUDA_VISIBLE_DEVICES=4 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty.yaml --gpus 0 --num_gpus 1 --name baseline/pertubesdf05 --perturb_sdf 0.5