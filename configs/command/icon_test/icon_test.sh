

# source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh
# conda activate /mnt/cephfs/home/qiuzhen/anaconda3/envs/icon3090v1/ 
conda activate icon3090v1
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

# CUDA_VISIBLE_DEVICES=6,7 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2 --test_code --mlp3d  --conv3d_start 3  #-test 
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2 #--test_code --mlpSe
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter.yaml -test --gpus 0 --test_code --mlpSe

# CUDA_VISIBLE_DEVICES=4 python -m apps.train -cfg configs/train/icon/icon-filter.yaml --gpus 0 --num_gpus 1 --res_layers 8 --name icon_wo_residual -test  #-test 
# CUDA_VISIBLE_DEVICES=4 python -m apps.train -cfg configs/train/icon/icon-filter.yaml --gpus 0 --num_gpus 1   --name baseline/icon_unet_5layer_v1 --use_unet  --mlp_dim 13 256 512 256 1 -test
CUDA_VISIBLE_DEVICES=3 python -m apps.train_and_eval -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 1   --name baseline/icon_smpl_noise_01 --resume #--test_code

# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8