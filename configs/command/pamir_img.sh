source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 


# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=1 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nf.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=1 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nf.yaml -test --gpus 0

CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml --gpus 0
CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml -test --gpus 0






# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg  configs/train/pamir_normal_debug/pamir_img_nf.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utiils_training/train_nerf.py -d 6



# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf_test.yaml --gpus 0
