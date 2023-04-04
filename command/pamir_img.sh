source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 

# python -m apps.train -cfg ./configs/train/pamir_img.yaml --gpus 3


# python -m apps.train -cfg ./configs/train/pamir_img.yaml -test --gpus 3

# python -m apps.train -cfg configs/train/pamir/pamir_img_nb_nf.yaml --gpus 3


# python -m apps.train -cfg configs/train/pamir/pamir_img_nb_nf.yaml -test --gpus 3

# python -m apps.train -cfg configs/train/pamir/pamir_img_nf.yaml --gpus 4


# python -m apps.train -cfg configs/train/pamir/pamir_img_nf.yaml -test --gpus 4

# python -m apps.train -cfg configs/train/pamir/pamir_img.yaml --gpus 6


# python -m apps.train -cfg configs/train/pamir/pamir_img.yaml -test --gpus 4
# python -m apps.train -cfg  configs/train/pamir/pamir_img_nb_nf.yaml -test --gpus 6



# python -m apps.train -cfg configs/train/pamir/pamir_img_nb.yaml -test --gpus 4
# python -m apps.train -cfg  configs/train/pamir/pamir_img_nf.yaml -test --gpus 4
# python -m apps.train -cfg configs/train/pamir/pamir_img.yaml -test --gpus 4
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utiils_training/train_nerf.py -d 6



python -m apps.train -cfg configs/train/pamir/pamir_img_nb_nf_test.yaml --gpus 3
