

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --resume


# CUDA_VISIBLE_DEVICES=2,3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wonorm.yaml --num_gpus 2 --gpus 0
# CUDA_VISIBLE_DEVICES=2,3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wonorm.yaml  -test --gpus 0

# CUDA_VISIBLE_DEVICES=3,2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wosdf.yaml --gpus 0 --num_gpus 2
# CUDA_VISIBLE_DEVICES=3,2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wosdf.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=0,1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wocmap.yaml --gpus 0 --num_gpus 2
# CUDA_VISIBLE_DEVICES=0,1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wocmap.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=3,2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wovis.yaml  --gpus 0 --num_gpus 2
CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wovis.yaml -test --gpus 0


python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8