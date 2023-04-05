## NORMAL MAP

**PaMir:**

Use pretrained normal network barely boost performance of huaman body reconstruction.

To be specific, see table below:

| Exp Name                                                 | col2                          | Group |                                                  |
| -------------------------------------------------------- | ----------------------------- | ----- | ------------------------------------------------ |
| pamir_using_img_only_epoch10_test                        | 1.85	1.52	0.11	1.94	1.58	0.11 | 1     |                                                  |
| pamir_using_img_only_epoch10_test_with_pretrained_Normal | 1.84	1.51	0.11	1.94	1.59	0.11 | 1     |                                                  |
| pamir_using_img_nb_nf_won                                | 1.62	1.35	0.11	1.66	1.36	0.1  | 2     |                                                  |
| pamir_using_img_nb_nf                                    | 1.62	1.36	0.11	1.67	1.37	0.1  | 2     |                                                  |
| pamir_using_img_normF_only_epoch10_won                   | 1.56	1.31	0.1	1.66	1.38	0.09  | 3     |                                                  |
| pamir_using_img_normF_only_epoch10                       | 1.57	1.31	0.1	1.65	1.37	0.09  | 3     |                                                  |
| pamir_v2_0220                                            | 1.48	1.16	0.1	1.86	1.41	0.11  | 4     | author's model which is trained with batchsize 8 |
|                                                          |                               |       |                                                  |
|                                                          |                               |       |                                                  |

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



---




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