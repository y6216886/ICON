## NORMAL MAP

**PaMir:**

Use pretrained normal network barely boost performance of huaman body reconstruction.

To be specific, see table below:

| Exp Name                                                 | cape-easy-chamfer cape-easy-p2s cape-easy-NC cape-hard-chamfer cape-hard-p2s cape-hard-NC | Group |                                                  |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----- | ------------------------------------------------ |
| pamir_using_img_only_epoch10_test                        | 1.85	1.52	0.11	1.94	1.58	0.11                                                             | 1     |                                                  |
| pamir_using_img_only_epoch10_test_with_pretrained_Normal | 1.84	1.51	0.11	1.94	1.59	0.11                                                             | 1     |                                                  |
| pamir_using_img_nb_nf_won                                | 1.62	1.35	0.11	1.66	1.36	0.1                                                              | 2     |                                                  |
| pamir_using_img_nb_nf                                    | 1.62	1.36	0.11	1.67	1.37	0.1                                                              | 2     |                                                  |
| pamir_using_img_normF_only_epoch10_won                   | 1.56	1.31	0.1	1.66	1.38	0.09                                                              | 3     |                                                  |
| pamir_using_img_normF_only_epoch10                       | 1.57	1.31	0.1	1.65	1.37	0.09                                                              | 3     |                                                  |
| pamir_v2_0220                                            | 1.48	1.16	0.1	1.86	1.41	0.11                                                              | 4     | author's model which is trained with batchsize 8 |
|                                                          |                                                                                           |       |                                                  |
|                                                          |                                                                                           |       |                                                  |

Different from PIFU that boosted more from *back normal map*, PaMir benefits more from *front normal map.*

---

For **pifu,** Use image only achieves the second best results, which is inferior to that counter. Moreover, the most distinguish feature is that it is not constrained by a SMPL model, thus the back normal map provide much information for the reconstruction process.

For pifu without norm exhibit better performance

| col1                     | col2                                           |                        | col3               |                                |
| ------------------------ | ---------------------------------------------- | ---------------------- | ------------------ | ------------------------------ |
| pifu_v1_0220             | 1.82	1.53	0.12	2.78	2.09	0.16                  |                        | authors model bs=8 | all with pretrained normal net |
| pifu_img+normb+normf     | 3.1	2.5	0.16	4.19	3.76	0.18                    | 0.07	1.05	1.06         | bs=2               |                                |
| pifu_img+normb+normf_won | 3.2883 2.3970 0.1739 4.4071 3.4281 0.2002 |                        |                    |                                |
| pifu_img+normb           | 3.48	2.31	0.17	4.44	3.39	0.2                   | 0.19	3.68	2.13         | bs=2               |                                |
| pifu_img+normb won       | 3.325 2.477 0.1649 4.291 3.475 0.187           |                        |                    |                                |
| pifu_img                 | 3.201 2.812 0.1792 4.406 4.102 0.1954     | 0.2010 3.4198 2.0241 | bs=2               |                                |
| pifu_img_won             | 3.236 2.544 0.1628 4.226 3.614 0.183           | 0.1858 3.466 2.409     |                    |                                |
| pifu_img+normf_won       | 3.1727 2.4459 0.1730 4.2307 3.5043 0.1927      | 0.185 3.317 2.111      |                    |                                |
| pifu_img+normf           | 4.23	2.31	0.22	5.06	3.15	0.24                  | 0.25	4.54	2.09         | bs=2               |                                |

| pifu_img+normb+normf_won | 3.2883 | 2.3970 | 0.1739 | 4.4071 | 3.4281 | 0.2002 |
| ------------------------ | -----: | -----: | -----: | -----: | -----: | -----: |
| pifu_img+normf_won       | 3.1727 | 2.4459 | 0.1730 | 4.2307 | 3.5043 | 0.1927 |

For ICON, using ground truth normal map and the normal net estimated by pretrained normal net

---

| Experimental name or path                                                                                | Details                                                                                                                                                                                                                                                       | group | cape-easy-chamfer cape-easy-p2s cape-easy-NC<br />cape-hard-chamfer cape-hard-p2s cape-hard-NCcol2 | chamfer p2s nc          |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------- | ----------------------- |
| [baseline/icon-filter_batch2_wo_debugv1](https://wandb.ai/y6216886/Human_3d_Reconstruction/runs/tyk4pzlx)  | without pretrained filer and normal net                                                                                                                                                                                                                      |       | 1.52 1.342 0.08451 1.523 1.377 0.08296                                                        |                         |
| data/results/baseline/icon-filter_batch2_withnormal_debugv1_0406                                         | without pretrained filter but with pretrained normal net                                                                                                                                                                                                      | 1     | 0.74020 0.73561 0.04851 0.87020 0.86693 0.05259                                                 | 0.8042 0.8699 0.06916   |
| data/results/baseline/icon-filter_batch2_withnormal_spatial_se_from_1st_layer                            | spatial squeeze and excitation module from 1st layer to second last layer                                                                                                                                                                                     |       | 0.76606 0.76051 0.05141 0.87931 0.87623 0.05476                                                    |                         |
| data/results/baseline/icon-filter_batch2_withnormal_mlpse                                                | without pretrained filter but with pretrained normal net<br />and spatial squeeze and excitation module on mlp 2st layer to second last layer                                                                                                                 | 1     | 0.85051 0.82118 0.06160 1.06506 1.01116 0.07027                                                    | 0.44903 0.46785 0.03745 |
| data/results/baseline/icon-filter_batch2_withnormal_mlpChannelSELayer                                    | without pretrained filter but with pretrained normal net<br />and channel squeeze and excitation module                                                                                                                                                       | 1     | 4.12281 2.64449 0.27502 4.10757 2.64847 0.26349                                                    | 2.51353 1.79984 0.16845 |
| data/results/baseline/icon-filter_batch2_withnormal_max(mlpse_mlpce)                                     | max(channel se , spatial se)                                                                                                                                                                                                                                  | 1     | 4.85704 1.89860 0.26063 4.85618 2.02986 0.25085                                                    | 3.02275 1.75938 0.19126 |
| icon-filter-v1_0220                                                                                      | using pretrained filer and ground truth normal map                                                                                                                                                                                                            |       | 0.7630 0.7255 0.0498 0.9025 0.8748 0.0523                                                     | 0.9935 0.9566 0.0805  |
| baseline/icon-filter_batch2_newresume                                                                    | mlp resume from "/mnt/cephfs/dataset/NVS/experimental_results<br />/avatar/icon/data/ckpt<br />/baseline/icon-filter_batch4_withnormal_debugv1/last.ckpt"<br />which only uses pretrained normal net to estimated normal net, <br />trained with batchsize 4 |       | 0.7957 0.778 0.05383 0.9414 0.9191 0.05817                                                         |                         |

| col1                           |                                                                                                                                                                                                                     | cape-easy-chamfer cape-easy-p2s cape-easy-NC cape-hard-chamfer cape-hard-p2s cape-hard-NCcol2 | CHAMFER P2S NC          | col3                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| full                           | without pretrained filter but with pretrained normal net                                                                                                                                                            | 0.74020 0.73561 0.04851 0.87020 0.86693 0.05259                                            |                         | data/results/baseline/icon-filter_batch2_withnormal_debugv1_0406                                                     |
| no vis feature                 |                                                                                                                                                                                                                     | 0.82729 0.84693 0.057028 0.942438 0.96122 0.058593                                       |                         | /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline/icon-filter_batch2_withnormal_wovis/  |
|                                |                                                                                                                                                                                                                     |                                                                                               |                         |                                                                                                                      |
| no cmap feature                | Removing this particular feature would result in a<br />slight increase in the performance of the ICON                                                                                                             | 0.72644 0.71112 0.0471799 0.87915 0.862821 0.052353                                      |                         |                                                                                                                      |
| **no sdf feature**       | It is quite an important feature!<br />Rerun this experiment to ensure                                                                                                                                              | 6.18129 1.01397 0.27234 7.91900 1.23011 0.33850                                               | 3.9583 0.6516 0.1790    | /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline/icon-filter_batch2_withnormal_wosdf   |
| **no sdf feature rerun** | After rechecking, the sdf feature indeed plays an important role in<br />boosting the reconstrucion performance, for its strong constraint on <br />the distance between points and SMPL (without cloth) surface. | 5.84651 1.07894 0.26394 7.66206 1.27595 0.32985                                               | 2.87665 0.71261 0.15532 | /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline/icon-filter_batch2_withnormal_wosdfv1 |
| no norm feature                |                                                                                                                                                                                                                     | 0.80651 0.79099 0.05373 1.00044 0.97546 0.06043                                               |                         | /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline/icon-filter_batch2_withnormal_wonorm  |

## SPEED

speed of code

for **pamir**

see .ICON/lib/net/HGPIFuNet.py  line 426

The code filter took 0.026144 seconds to run.
The code query took 0.019207 seconds to run.
The code get_error took 0.000508 seconds to run.

torch.Size([2, 3, 8000]) point size
The code extract volume feature took 0.008257 seconds to run.
The code feature select took 0.000223 seconds to run.
torch.Size([2, 13, 8000]) point_feat size
The code query took 0.001276 seconds to run.

for **ICON**

The code filter took 0.034700 seconds to run.
The code query took 5.242476 seconds to run.
The code get_error took 0.000279 seconds to run.

torch.Size([2, 3, 8000]) point size
The code extract volume feature took 7.217828 seconds to run.
The code feature select took 0.000940 seconds to run.

torch.Size([2, 13, 8000]) point_feat size
The code query took 0.002476 seconds to run.

The query process takes too much time for icon!
