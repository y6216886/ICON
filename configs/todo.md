pamir with (1) img (2) img+normalf (3)img+normalb (4) img+normalf+normalb  gpu025

pifu with (1) img (2) img+normalf (3)img+normalb (4) img+normalf+normalb  gpu025

use pretrained normal network boost performance a bit, not too much

baseline/pifu_img+normf_won  oom

add orientation map to reconstruction

study the effects of normal net

evaluate (1) pifu image won (2) img+norf+wn  (3) img+norb+won

pifu image encoder to triplane
pamir mesh voxelization to triplane
icon how to translate smpl to sdf

test time adaptation mseloss no entropy
modify loss

mask auto encoder
add a discriminator to discriminate between the real 3d model and the false 3d model?
dense prediction of sdf

note: in get normal net() function    the weight of normal net is fixed.!!  so normal net is not updated
evaluate (1) icon won bs2 (2) icon wn bs2  2023 0405   gpu021

gpu027  1) icon bs=4 2) icon bs=8
gpu027  pifu  1) image + nf +nb  2) image + nf 3)image + nb; all with estimated normal map
gpu024 pamir 1) image + nf +nb  2) image + nf 3)image + nb; all with estimated normal map

0407 gpu025 icon with pretrained noraml without pretrained filter
icon without normal performance is too good, figure out why
  1）see gpu022 我用了一个没训练完的icon ckpt 看看这个性能如何
  2）

todo: save codes configs

todo: study the impact of smpl features vis cmap norm and sdf

todo: use part segmentation map to extract feature for mlp  (see body net fig 2)

---

* Evaluate time cost and gpu memory cost of different gpu setting by varing batch size

gpu026 ddp #gpu=2  bs=2 -> 1.9s/iteration 2 hours/epoch  gpu memory 8732MiB/per-gpu use this one

gpu026 ddp #gpu=2  bs=8 -> 1.9s/iteration 2h:40m /epoch  gpu memory 17000MiB

gpu026 ddp #gpu=2  bs=4 -> 1.9s/iteration 2h:40m /epoch  gpu memory 15000MiB

gpu026 without_ddp #gpu=1  bs=2 -> 1.9s/iteration 3h:20m /epoch  gpu memory 7000MiB

* Use ddp=2 bs=2

---

### Study feature

I set pytorch lightning trainer with profile = "pytorch" to check the bottleneck of the whole training process in gpu026 bottom left 2 window.

ablation on smpl feature

study the dim of each dim

0410 gpu025 gpu026 study feature

0411 try to remove bary weight

0413 mlp se block     channel se and spatial se  gpu026

0413 mlp se block torch.max(c_se, s_se)

0415 mixermlp with chunk

0417 gpu023 gpu025 gpu026  conv first layer 1-4

0418 gpu021 conv3d zeros conv5 conv7

gpu025 conv3d start from 1 2 3 ##the point position are random, is it reasonable to use conv3d?

gpu025 check icon baseline 0419

gpu018 check without cmap

gpu017  mlp3d conv3d start from 0  (1)with cmap  (2)without cmap

gpu017 convkernel=3, mlp3d conv3d start from 0,1,2  ##kernel size >3 leads to collapse error

gpu017 convkernel=1 without using cmap

gpu023 icon without batchnorm

gpu022 unet mlp 2023 0422
gpu016 dropout 0.1  gpu017 dropout 0.2

gpu021 icon_pamir se sev1

smplx noise num_worker8 6hr for an epoch 8G GPU MEMORY
smplx noise num_worker1 120hr for an epoch 8G GPU MEMORY
smplx noise num_worker32 6hr for an epoch

0507
gpu020 icon pamir mlpse 0.1 and 1
pu024 icon pamir 
gpu023 pamir
---
### N net MLP i.e., converse U net

### How to extend the 3D human digitilization task to embrace the trends of data centric AI?

    Getting more image to occupancy label pairs

    Improving the quality of existing data

### How to modify MLP for nerf, 3d human digitalization, etc.

### Single view synthesis for human body xxx

### Uncertainty during training

image feature from clip , concate it with input of mlp

### Fine granularity (e.g., face, hands, hair, foot) details are far from promising

### Depth feature

depth feature is the most important feature for mlp input, try to use kl loss to align the distribution of occupancy and smpl depth?

### aumentation methods like augnerf 
  1.aug sdf  ####has already been done
  2.aug normal map
  