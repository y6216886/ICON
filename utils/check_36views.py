path='/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2_36views/'
import os
# subjectlist=os.listdir(path)
# print(subjectlist)

import glob
subjectlist=glob.glob(path+'*/vis/*')
print(len(subjectlist))



smpl_type = "smplx" if (
            'smplx_path' in data_dict.keys() and os.path.exists(data_dict['smplx_path'])
        ) else "smpl"
return_dict = {}

if 'smplx_param' in data_dict.keys() and \
    os.path.exists(data_dict['smplx_param']) and \
        sum(self.noise_scale) > 0.0:
    smplx_verts, smplx_dict = self.compute_smpl_verts(
        data_dict, self.noise_type, self.noise_scale
    )
    smplx_faces = torch.as_tensor(self.smplx.smplx_faces).long()
    smplx_cmap = torch.as_tensor(np.load(self.smplx.cmap_vert_path)).float()

else:
    smplx_vis = torch.load(data_dict['vis_path']).float()
    return_dict.update({'smpl_vis': smplx_vis})