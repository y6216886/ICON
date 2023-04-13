

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_whole -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2 --test_code
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml --gpus 0 --num_gpus 2  --mlpSe
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml -test --gpus 0 --mlpSe

CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml --gpus 0 --num_gpus 2  --mlpSev1
CUDA_VISIBLE_DEVICES=3 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml -test --gpus 0 --mlpSev1


CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8