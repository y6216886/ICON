#  source  /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh
# conda env create -f environment.yaml

# # conda init bash
# # source ~/.bashrc
# # source activate icon3090v1
# conda activate icon3090v1
# https_proxy=socks5h://cpu0 pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
https_proxy=socks5h://cpu0 MAX_JOBS=1 PYTORCH3D_NO_NINJA=1 pip install -r requirements.txt --use-deprecated=legacy-resolver
# https_proxy=socks5h://cpu0 MAX_JOBS=1 PYTORCH3D_NO_NINJA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
 

conda activate   /mnt/cephfs/home/qiuzhen/anaconda3/envs/icon3090v1
https_proxy=socks5h://cpu0 PIP_CACHE_DIR=/mnt/cephfs/home/qiuzhen/ pip install wandb                                                
 ##note .cache space is not enough thus 
 # use PIP_CACHE_DIR to change a path