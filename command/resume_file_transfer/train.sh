

# source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# # conda activate icontitanx ##icon for 3090
# conda activate icon3090
# cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

# CUDA_VISIBLE_DEVICES=5 python -m apps.train -cfg configs/train/icon/icon-filter.yaml -test --gpus 0 #--name icon-filter_batch2_withnormal_debugv1_0417



# CUDA_VISIBLE_DEVICES=4,5 python -m apps.train_and_eval -cfg configs/train/icon_unet/icon-filterunet.yaml --gpus 0 --num_gpus 2  --mlp_dim 13 128 256 512 256 128 1 --res_layers 2 3 4 5 6 --name baseline/icon_unetv1 --use_unet
# CUDA_VISIBLE_DEVICES=4,5 python -m apps.train -cfg configs/train/icon_unet/icon-filterunet.yaml --gpus 0 --num_gpus 2  --mlp_dim 13 128 256 512 256 128 1 --res_layers 2 3 4 5 6 --name baseline/icon_unetv1 --use_unet -test
CUDA_VISIBLE_DEVICES=0 python -m apps.train_and_eval -cfg command/resume_file_transfer/icon.yaml --gpus 0 --num_gpus 1   --name baseline/icon-filter_batch2_newresumev1 --resume #-test --mlp_dim 13 128 256 512 256 128 1 --res_layers 2 3 4 5 6
