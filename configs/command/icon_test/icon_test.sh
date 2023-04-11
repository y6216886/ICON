

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2 --test_code
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/icon/icon-filter.yaml -test --gpus 0 --test_code



# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8