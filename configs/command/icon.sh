

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=4,5 python -m apps.train -cfg configs/train/icon/icon-filter.yaml -test --gpus 0 --num_gpus 1 #--name icon-filter_batch2_withnormal_debugv1_0417


# python -m apps.train -cfg configs/train/icon/icon-filter_bs8.yaml --gpus 0

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --test


# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter_wo.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter_wo.yaml --gpus 0 -test



# CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon/icon-filter_bs4.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon/icon-filter_bs4.yaml --gpus 0 -test

# CUDA_VISIBLE_DEVICES=1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon/icon-filter_bs8.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon/icon-filter_bs8.yaml --gpus 0 -test



# CUDA_VISIBLE_DEVICES=1,2 python -m apps.train_and_eval -cfg configs/train/icon/icon-filter_withoutBN.yaml --gpus 0 --num_gpus 2 --name withoutBN


CUDA_VISIBLE_DEVICES=2,4 python -m apps.train_and_eval -cfg configs/train/icon/icon-filter.yaml --gpus 0 --num_gpus 2  --name icon_dropout
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 0 2 3 --occ 0.8