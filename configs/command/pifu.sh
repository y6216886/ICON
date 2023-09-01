# source /mnt/cephfs/home/yangyifan/miniconda/etc/profile.d/conda.sh
# # conda activate icontitanx ##icon for 3090
# conda activate icon3090;cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/ 


# CUDA_VISIBLE_DEVICES=3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normf.yaml --gpus 0
# CUDA_VISIBLE_DEVICES=3 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normf.yaml -test --gpus 0



# # CUDA_VISIBLE_DEVICES=2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normb.yaml --gpus 0
# # CUDA_VISIBLE_DEVICES=2 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normb.yaml -test --gpus 0




# # CUDA_VISIBLE_DEVICES=1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normb+normf.yaml --gpus 0
# # CUDA_VISIBLE_DEVICES=1 WANDB__SERVICE_WAIT=300 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img+normb+normf.yaml -test --gpus 0

# python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py -d 3 --occ 0.8



# # CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img.yaml --gpus 0
# # CUDA_VISIBLE_DEVICES=0 python -m apps.train -cfg  configs/train/pifu_normal_debug/pifu_img.yaml -test --gpus 0

for i in {1..10}
do
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh; conda activate icon3090v1; cd /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/; CUDA_VISIBLE_DEVICES=6 python -m apps.train -cfg  /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/configs/train/pifu/pifu_img.yaml --gpus 0 --name trainonthuman2/1train_on_thuman2_36views_pifu_v11{$i} --datasettype thuman2  --num_worker 4  --logger tb  #trainonthuman2
done