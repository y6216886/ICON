

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 




# CUDA_VISIBLE_DEVICES=3 python -m apps.train -cfg configs/train/icon_uncertainty/icon-filter_uncertainty003.yaml --gpus 0 --num_gpus 1  --uncertainty --beta_min 0.03 --beta_plus 3 --test_mode

# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_uncertainty/icon-filter_uncertainty003.yaml --gpus 0 --num_gpus 2  --uncertainty --beta_min 0.03 --beta_plus 3 #--test_code
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 0 2 3 --occ 0.8

