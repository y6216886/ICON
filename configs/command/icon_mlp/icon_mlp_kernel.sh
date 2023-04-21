

source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# conda activate icontitanx ##icon for 3090
conda activate icon3090
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10

###parser.add_argument(--mlpSe, default=False, action=store_true) ##spatial se
###parser.add_argument(--mlpSev1, default=False, action=store_true) ##channel se


# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml --gpus 0 --num_gpus 2  --mlpSe
# CUDA_VISIBLE_DEVICES=5 python -m apps.train -cfg configs/train/icon_mlp/icon-filter.yaml -test --gpus 0 --mlpSe


# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_convse_from_1st_layer.yaml --gpus 0 --num_gpus 2 --mlpSev1 ##--test_code
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer.yaml --gpus 0 --num_gpus 2 --mlpSe ##--test_code
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml --gpus 0 --num_gpus 2  --mlpSev1
# CUDA_VISIBLE_DEVICES=4 python -m apps.train -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml -test --gpus 0 --mlpSev1

# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_max.yaml --gpus 0 --num_gpus 2 --mlpSemax 
# CUDA_VISIBLE_DEVICES=1,0 python -m apps.train -cfg configs/train/icon_mlp/icon-filter_max.yaml --gpus 0 --num_gpus 1 --mlpSemax  -test
# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filterChannelSELayer.yaml --gpus 0 --num_gpus 2 --mlpSev1 




# CUDA_VISIBLE_DEVICES=6,7 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer_conv3d_1.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 1 ##--test_code 
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer_conv3d_2.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 2
# CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer_conv3d_3.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 3
# CUDA_VISIBLE_DEVICES=1,2 python -m apps.train_and_eval -cfg configs/train/icon_mlp/icon-filter_spatialse_from_1st_layer_conv3d_4.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4

# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel3_paddmode_replicate.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 3 --pad_mode replicate    #--test_code 
# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel5_paddmode_replicate.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 5 --pad_mode replicate #--test_code 
# CUDA_VISIBLE_DEVICES=0,1 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel7_paddmode_replicate.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 7 --pad_mode replicate #--test_code 

# CUDA_VISIBLE_DEVICES=1,2 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel3_paddmode_zeros.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 3 --pad_mode zero    #--test_code 
# CUDA_VISIBLE_DEVICES=1,6 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel5_paddmode_zeros.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 5 --pad_mode zero #--test_code 
# CUDA_VISIBLE_DEVICES=1,6 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d_start4_kernel7_paddmode_zeros.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 4 --conv3d_kernelsize 7 --pad_mode zero #--test_code 

CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 3 --conv3d_kernelsize 3 --pad_mode zero --name baseline/3dmlp/convstart3_kernel3_padzero  --test_code 

CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 2 --conv3d_kernelsize 3 --pad_mode zero  --name baseline/3dmlp/convstart2_kernel3_padzero --test_code 

CUDA_VISIBLE_DEVICES=2,3 python -m apps.train_and_eval -cfg configs/train/icon_mlp/3dmlp_kernel/icon-filter_withnormal_conv3d.yaml --gpus 0 --num_gpus 2 --mlp3d --conv3d_start 0 --conv3d_kernelsize 3 --pad_mode zero  --name baseline/3dmlp/convstart0_kernel3_padzero_withoutcmap --test_code 



CUDA_VISIBLE_DEVICES=1,2 python -m apps.train -cfg configs/train/icon/icon-filter_test.yaml --gpus 0 --num_gpus 2
# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 2 3 --occ 0.8