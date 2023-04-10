source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# python -m scripts.render_batch -headless -out_dir data/
# https_proxy=socks5h://cpu0 pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# https_proxy=socks5h://cpu0 pip install -r requirements.txt --use-deprecated=legacy-resolver

# python -m scripts.render_batch -headless -out_dir data/
# CUDA_VISIBLE_DEVICES=4 python -m scripts.visibility_batch -out_dir data   #gpu025


# python -m lib.dataloader_demo -v -c ./configs/train/icon-filter.yaml
# python -m lib.dataloader_demo -v -c ./configs/train/pamir.yaml
# cephdu -d 1 -h /mnt/cephfs/home/yangyifan/yangyifan | sort -h
# Training for implicit MLP
# python -m apps.train -cfg ./configs/train/icon-filter.yaml --gpus 1
python -m apps.train -cfg ./configs/train/pamir.yaml --gpus 6

# Training for normal network


# Training for implicit MLP
# python -m apps.train -cfg ./configs/train/pifu.yaml --gpus 0

# python -m apps.train -cfg ./configs/train/icon-filter.yaml -test  --gpus 1
python -m apps.train -cfg ./configs/train/pamir.yaml -test --gpus 6
# Training for normal network
# CUDA_VISIBLE_DEVICES=6 python -m apps.train-normal -cfg ./configs/train/normal.yaml

# Training for implicit MLP
# python -m apps.train -cfg ./configs/train/pifu.yaml -test --gpus 0

# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/0029/vis/290.pt '0029', '0210', '0311', '0410', '0511'
# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/0210/vis/130.pt
# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/0311/vis/090.pt
# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/0410/vis/310.pt
# /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/0511/vis/340.pt
