
source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 

python -m apps.train -cfg configs/train/pamir/pamir_img_nb.yaml --gpus 3


python -m apps.train -cfg configs/train/pamir/pamir_img_nb.yaml -test --gpus 3

