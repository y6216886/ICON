
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh
# conda activate /mnt/cephfs/home/qiuzhen/anaconda3/envs/icon3090v1/ 
conda activate icon3090v1
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 




# CUDA_VISIBLE_DEVICES=3 python -m apps.train -cfg configs/train/icon_uncertainty/icon-filter_uncertainty003.yaml --gpus 0 --num_gpus 1  --uncertainty --beta_min 0.03 --beta_plus 3 --test_mode

# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty003.yaml --gpus 0 --num_gpus 2  --uncertainty --beta_min 0.03 --beta_plus 3 #--test_code
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 0 2 3 --occ 0.8

# CUDA_VISIBLE_DEVICES=6 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertaintyv1.yaml --gpus 0 --num_gpus 1  --uncertainty --name baseline/uncertainty_logv1_pamir_icon --pamir_icon   #--test_code

# CUDA_VISIBLE_DEVICES=0 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertaintyv1.yaml --gpus 0 --num_gpus 1  --uncertainty --name baseline/uncertainty_logv1_pamir_icon --pamir_icon --mlp_first_dim 45

# CUDA_VISIBLE_DEVICES=2 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertaintyv1.yaml --gpus 0 --num_gpus 1  --uncertainty --name baseline/uncertainty_logv1_noise01 --noise_scale 0.1 0.1

# CUDA_VISIBLE_DEVICES=6 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertaintyv1.yaml --gpus 0 --num_gpus 1  --uncertainty --name baseline/uncertainty_logv1_pamir_icon_noise01 --noise_scale 0.1 0.1 --pamir_icon --mlp_first_dim 45

# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 0 2 3 --occ 0.8

CUDA_VISIBLE_DEVICES=6 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertaintyv1.yaml --gpus 0 --num_gpus 1  --uncertainty --name baseline/uncertainty_logv1_kl --kl_div
