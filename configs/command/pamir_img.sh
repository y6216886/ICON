source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh
# conda activate /mnt/cephfs/home/qiuzhen/anaconda3/envs/icon3090v1/ 
conda activate icon3090v1
cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 
# export OMP_NUM_THREADS=10


# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=1 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nf.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=1 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nf.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml -test --gpus 0

# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml -test --gpus 0






# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg  configs/train/pamir_normal_debug/pamir_img_nf.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img.yaml -test --gpus 0
# CUDA_VISIBLE_DEVICES=2 python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utiils_training/train_nerf.py -d 6



# CUDA_VISIBLE_DEVICES=2 python -m apps.train -cfg configs/train/pamir_normal_debug/pamir_img_nb_nf_test.yaml --gpus 0

# CUDA_VISIBLE_DEVICES=1 python -m apps.train_and_eval -cfg configs/train/pamir/pamir_img_noise.yaml --gpus 0 --num_gpus 1 --name baseline/pamir_perturb_smpl_1  

# CUDA_VISIBLE_DEVICES=5 python -m apps.train_and_eval -cfg configs/train/pamir/pamir_img_noise.yaml --gpus 0 --num_gpus 1 --name baseline/pamir_perturb_smpl_01

# CUDA_VISIBLE_DEVICES=5 python -m apps.train_and_eval -cfg configs/train/pamir/pamir_img_noise.yaml --gpus 0 --num_gpus 1 --name baseline/pamir_perturb_smpl_001v1  --noise_scale 0.01 0.01

# CUDA_VISIBLE_DEVICES=2 python -m apps.train_and_eval -cfg configs/train/pamir/pamir_img_noise.yaml --gpus 0 --num_gpus 1 --name baseline/pamir_perturb_smpl_05v1 

CUDA_VISIBLE_DEVICES=5 python -m apps.train_and_eval -cfg configs/train/pamir/pamir_img_noise.yaml --gpus 0 --num_gpus 1 --name baseline/pamir_perturb_smpl_02  --noise_scale 0.2 0.2