

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

###parser.add_argument(--mlpSe, default=False, action=store_true) ##spatial se
###parser.add_argument(--mlpSev1, default=False, action=store_true) ##channel se


# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml --gpus 0 --num_gpus 2  --mlpSe
# CUDA_VISIBLE_DEVICES=5 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml -test --gpus 0 --mlpSe


# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_convse_from_1st_layer.yaml --gpus 0 --num_gpus 2 --mlpSev1 #--test_code
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer.yaml --gpus 0 --num_gpus 2 --mlpSe #--test_code
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml --gpus 0 --num_gpus 2  --mlpSev1
# CUDA_VISIBLE_DEVICES=4 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml -test --gpus 0 --mlpSev1

# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_max.yaml --gpus 0 --num_gpus 2 --mlpSemax 
# CUDA_VISIBLE_DEVICES=1,0 python -m apps.train -cfg configs/train/icon_mlp/icon-filter_max.yaml --gpus 0 --num_gpus 1 --mlpSemax  -test
# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml --gpus 0 --num_gpus 2 --mlpSev1 




CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_mlp3d_mixermlp.yaml --gpus 0 --num_gpus 2 --mlp3d #--test_code 
CUDA_VISIBLE_DEVICES=0,1 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8