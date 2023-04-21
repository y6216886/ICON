

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# gpu=0,1
num_gpu=2
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --resume


# CUDA_VISIBLE_DEVICES=0,1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wonorm.yaml --num_gpus $num_gpu --gpus 0 --mlp_first_dim 10 #--test_code
# CUDA_VISIBLE_DEVICES=2,3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wonorm.yaml  -test --gpus 0 --mlp_first_dim 10 #--test_code

# CUDA_VISIBLE_DEVICES=2,3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wosdf.yaml --gpus 0 --num_gpus 2  --mlp_first_dim 12   ##have 12 channels
# CUDA_VISIBLE_DEVICES=1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wosdf.yaml -test --gpus 0 --mlp_first_dim 12 

CUDA_VISIBLE_DEVICES=1,4 WANDB__SERVICE_WAIT=300 python -m apps.train_and_eval -cfg configs/train/icon_study_feature/icon-filter_wocmap.yaml --gpus 0 --num_gpus 2 --mlp_first_dim 10 --mlp3d --conv3d_start 0 --conv3d_kernelsize 1 --pad_mode zeros --name mlp3d_convstart0_kernel1_padzeros_wocmap  #have 10 channels
# CUDA_VISIBLE_DEVICES=3,2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wocmap.yaml -test --gpus 0 --mlp_first_dim 10

# CUDA_VISIBLE_DEVICES=3,2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wovis.yaml  --gpus 0 --num_gpus 2
# CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg configs/train/icon_study_feature/icon-filter_wovis.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus $num_gpu

# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8