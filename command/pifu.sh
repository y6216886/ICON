source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 



# python -m apps.train -cfg  configs/train/pifu/pifu_img.yaml --gpus 0
# python -m apps.train -cfg  configs/train/pifu/pifu_img.yaml -test --gpus 3

# python -m apps.train -cfg  configs/train/pifu/pifu_img+normb.yaml --gpus 1
# python -m apps.train -cfg  configs/train/pifu/pifu_img+normb.yaml -test --gpus 1

python -m apps.train -cfg  configs/train/pifu/pifu_img+normf.yaml --gpus 1
python -m apps.train -cfg  configs/train/pifu/pifu_img+normf.yaml -test --gpus 1


# python -m apps.train -cfg  configs/train/pifu/pifu_img+normb+normf.yaml --gpus 7
# python -m apps.train -cfg  configs/train/pifu/pifu_img+normb+normf.yaml -test --gpus 7

# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utiils_training/train_nerf.py -d 6
