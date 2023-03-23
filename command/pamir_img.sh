source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 

python -m apps.train -cfg ./configs/train/pamir_img.yaml --gpus 5


python -m apps.train -cfg ./configs/train/pamir_img.yaml -test --gpus 5

