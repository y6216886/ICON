conda activate icon3090v1
cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/reproduce/train11_on_cape_vol3_pe17_mlpse_start_c_1_end_c_3_spatialse_r_APE_smpl_attention_wodetach{2}1/codes_infer; CUDA_VISIBLE_DEVICES=1  python -m apps.inferv1 -cfg /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/configs/train/icon_uncertainty/icon-filter_uncertainty003.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 10 --pamir_vol_dim 6 --se_end_channel 3 --se_reduction 4 --se_start_channel 1  --smpl_attention  -gpu 0 -loop_smpl 100 -loop_cloth 200 -hps_type pymaf --ckpt_path /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/ckpt/trainonthuman/train11_on_thuman_vol{6}_pe{10}_mlpse_start_c_1_end_c_3_spatialse_r_APE_smpl_attentionb1/last.ckpt   --adaptive_pe_sdf --expname sketch -in_dir $rgb_path -out_dir $obj_parent_path
