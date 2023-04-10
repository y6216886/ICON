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

I set pytorch lightning trainer with profile = "pytorch" to check the bottleneck of the whole training process in gpu026 bottom left 2 window.

ablation on smpl feature

study the dim of each dim
