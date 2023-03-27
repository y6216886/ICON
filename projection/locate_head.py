import numpy as np
import cv2
import pandas as pd
import torch
def read_vertex_faces_id_from_obj(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res_f = []
    res_v = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res_f.append(v)
            elif line.startswith('v '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [float(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [float(i.strip('\n')) for i in tmp[1:4]]
                res_v.append(v)
    res_f_list=np.array(res_f, dtype=np.int) - 1 
    res_v_list=np.array(res_v, dtype=np.float32)
    return res_f_list, res_v_list# obj index from 1

def write_numpy_to_obj(verts, filename=None):
    thefile = open('./'+filename, 'w')
    for item in verts:
      a,b,c=item[0].item(),item[1].item(),item[2].item()
      thefile.write("v {0} {1} {2}\n".format(a,b,c))

    # for item in normals:
    #   thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    # for item in faces:
    #   thefile.write("f {0} {1} {2}\n".format(item[0],item[1],item[2]))  

    thefile.close()

def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    # points=points[None,...]
    # calibrations=calibrations[None,...]
    rot = calibrations[:3, :3]
    trans = calibrations[:3, 3:4]
    # pts = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    pts = torch.mm(rot, points.T).T 
    pts+= (trans.T).repeat(pts.size(0),1)
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    # calib_mat = np.matmul(intrinsic, extrinsic)
    extrinsic = torch.from_numpy(extrinsic).float()
    intrinsic = torch.from_numpy(intrinsic).float()
    return extrinsic, intrinsic

def load_calib_togther(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()

    return calib_mat

def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]

def get_face_idx_from_smplx(smplx_obj_numpy,smplx_flame_vertex_ids="projection/SMPL-X__FLAME_vertex_ids.npy"):
    sf_ids = np.load(smplx_flame_vertex_ids)
    smplx_obj_numpy=np.array(smplx_obj_numpy)
    smplx_vertex_numpy=list(list(i) for i in smplx_obj_numpy[sf_ids])

    return smplx_vertex_numpy

    
def homogeneous_2d(point_2d): 
    point_2d[0] = point_2d[0]/point_2d[2]
    point_2d[1] = point_2d[1]/point_2d[2]
    point_2d[2] = point_2d[2]/point_2d[2]
    return point_2d

# def reproject(projectionMatrix_ex, point_3d):
#     points_2d = np.empty((point_3d.shape[0],3))
#     ones = np.ones((point_3d.shape[0],1))
#     point_3d = np.hstack((point_3d,ones))

#     for i in range(point_3d.shape[0]):
#         reprojected = np.matmul(projectionMatrix_ex,point_3d[i])
#         homogeneous_2d(reprojected)
#         points_2d[i][0] = reprojected[0]
#         points_2d[i][1] = reprojected[1]
#         points_2d[i][2] = reprojected[2]

#     return points_2d[:,:2]

def reproject(projectionMatrix_ex, projectionMatrix_in, point_3d):
    points_2d = np.empty((point_3d.shape[0],3))
    # ones = np.ones((point_3d.shape[0],1))
    # point_3d = np.hstack((point_3d,ones))

    for i in range(point_3d.shape[0]):
        # reprojected = np.matmul(projectionMatrix_ex,point_3d[i])
        Rotation=projectionMatrix_ex[:3, :3]
        Trans=projectionMatrix_ex[:3, 3]
        points_t=point_3d[i].T
        rotated_points_t=np.matmul(Rotation, points_t).T
        reprojected= rotated_points_t + Trans
        reprojected=homogeneous_2d(reprojected)
        reprojected=np.matmul(projectionMatrix_in, reprojected)
        points_2d[i][0] = reprojected[0]
        points_2d[i][1] = reprojected[1]
        points_2d[i][2] = reprojected[2]

    return points_2d[:,:2]


def uv_of_face_in_image_reproject(body_obj_path="projection/0001.obj", calib_path="projection/120.txt", image_path='projection/120.png'):
    smpl_whole_body=read_vertex_faces_id_from_obj(body_obj_path)
    vertex_whole_body = smpl_whole_body[1]
    face_smplx_vertex_numpy=get_face_idx_from_smplx(vertex_whole_body)
    face_smplx_vertex_torch=torch.Tensor(face_smplx_vertex_numpy)* 100.0

    calib=load_calib_togther(calib_path)
    transformed_face=orthogonal(face_smplx_vertex_torch, calib)
    write_numpy_to_obj(transformed_face, filename="face.obj")
    transformed_face=(transformed_face+1)/2*512
    img = cv2.imread(image_path)
    for i in range(transformed_face.shape[0]):
            joint_img = cv2.circle(img, (int( transformed_face[i][0]), int(transformed_face[i][1])), 1, (0,255,0),-1)
    cv2.imwrite(image_path.split(".")[0]+"_projected.png", joint_img)

import os
from PIL import Image
def uv_of_face_in_image_crop(body_obj_path='projection/0001.obj', calib_path="projection/120.txt", image_path='projection/120.png'):
    smpl_whole_body=read_vertex_faces_id_from_obj(body_obj_path)
    vertex_whole_body = smpl_whole_body[1]
    face_smplx_vertex_numpy=get_face_idx_from_smplx(vertex_whole_body)
    face_smplx_vertex_torch=torch.Tensor(face_smplx_vertex_numpy)* 100.0

    calib=load_calib_togther(calib_path)
    transformed_face=orthogonal(face_smplx_vertex_torch, calib)
    # write_numpy_to_obj(transformed_face, filename="face.obj")
    transformed_face=(transformed_face+1)/2*512
    x_min= max(int(torch.min(transformed_face[:,0]))-30,0)
    x_max= min(int(torch.max(transformed_face[:,0]))+30,512)
    y_min= max(int(torch.min(transformed_face[:,1]))-30,0)
    y_max= min(int(torch.max(transformed_face[:,1]))+60,512)
    img = Image.open(image_path).convert("RGBA")
    # img_numpy_mask=np.array(img)[...,3]
    # img_numpy_mask[img_numpy_mask>0]=255
    # Image.fromarray(img_numpy_mask).save("projection/mask.png")
    # print(np.unique(img_numpy_mask))

    # Newimg=Image.new("RGBA",img.size,"GREEN")
    # Newimg.paste(img,mask=img)
    # Newimg=Newimg.crop([x_min,y_min,x_max,y_max])
    Newimg=img.crop([x_min,y_min,x_max,y_max])
    Newimg.save("projection/"+os.path.split(image_path)[1].split('.')[0]+"_cropv2.png")
    print("synthesized to path.{}".format("projection/"+os.path.split(image_path)[1].split('.')[0]+"_cropv2.png"))

    # joint_img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),color=(0,255,0))
    # crop_img = cv2.crop(img,(x_min,y_min),(x_max,y_max))
    # cv2.imwrite("projection/"+os.path.split(image_path)[1].split('.')[0]+"_crop.png", crop_img)

# def uv_of_face_in_image(body_obj_path='projection/0001.obj', calib_path="projection/120.txt", image_path='projection/120.png'):
#     smpl_whole_body=read_vertex_faces_id_from_obj(body_obj_path)
#     vertex_whole_body = smpl_whole_body[1]
#     face_smplx_vertex_numpy=get_face_idx_from_smplx(vertex_whole_body)
#     face_smplx_vertex_torch=torch.Tensor(face_smplx_vertex_numpy)* 100.0

#     calib=load_calib_togther(calib_path)
#     transformed_face=orthogonal(face_smplx_vertex_torch, calib)
#     # write_numpy_to_obj(transformed_face, filename="face.obj")
#     transformed_face=(transformed_face+1)/2*512
#     x_min= max(int(torch.min(transformed_face[:,0]))-30,0)
#     x_max= min(int(torch.max(transformed_face[:,0]))+30,512)
#     y_min= max(int(torch.min(transformed_face[:,1]))-30,0)
#     y_max= min(int(torch.max(transformed_face[:,1]))+60,512)
#     img = cv2.imread(image_path)
#     joint_img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),color=(0,255,0))
#     crop_img = cv2.crop(img,(x_min,y_min),(x_max,y_max))
#     cv2.imwrite("projection/"+os.path.split(image_path)[1].split('.')[0]+"_crop.png", crop_img)



if __name__ =="__main__":
    kwargs={"body_obj_path":'projection/0001.obj',
       "calib_path":"projection/120.txt",
       "image_path":"projection/120.png",
       }
    kwargsv1={"body_obj_path":'projection/0000.obj',
       "calib_path":"projection/000.txt",
       "image_path":"projection/000.png",
       }
    kwargsv2={"body_obj_path":'projection/0260.obj',
       "calib_path":"projection/040.txt",
       "image_path":"projection/040.png",
       }
    kwargsv3={"body_obj_path":'data/thuman2/smplx/0016.obj',
       "calib_path":"data/thuman2_36views/0016/calib/010.txt",
       "image_path":"data/thuman2_36views/0016/render/010.png",
       }

    
    # uv_of_face_in_image_crop(**kwargsv3) #reproject vertex of face into image
    # uv_of_face_in_image_crop(**kwargs)
    smpl_whole_body=read_vertex_faces_id_from_obj("projection/0000.obj")
    vertex_whole_body = smpl_whole_body[1]
    smpl_whole_body=list(list(i) for i in vertex_whole_body)


 
    # vertex_whole_body=torch.from_numpy(vertex_whole_body)* 100.0
    # calib=load_calib("/home/young/code/HairNet_orient2D/maskrcnn_segmentation/locate_head_with_smpl/000.txt")
    # transformed_body=projection(vertex_whole_body, calib)
    # transformed_body[:, 1] *= -1

    # write_numpy_to_obj(transformed_body, filename="body.obj")

    smplx_vertex_numpy=get_face_idx_from_smplx(smpl_whole_body)
    face_smplx_vertex_torch=torch.Tensor(smplx_vertex_numpy)* 100.0
    calib=load_calib_togther("projection/000.txt")
    transformed_face=orthogonal(face_smplx_vertex_torch, calib)
    print(torch.max(transformed_face), torch.min(transformed_face))
    # face_smplx_vertex_torch[:, 1] *= -1
    write_numpy_to_obj(transformed_face, filename="body.obj")
    transformed_face=(transformed_face+1)/2*512
    # transformed_face_v1 = torch.div(transformed_face[:,:2] , depth)
    # gt_joint = reproject(calib_ex,calib_in, face_smplx_vertex_torch)*512 ##2d
    img = cv2.imread('projection/000.png')
    for i in range(transformed_face.shape[0]):
            joint_img = cv2.circle(img, (int( transformed_face[i][0]), int(transformed_face[i][1])), 1, (0,255,0),-1)
    cv2.imwrite('projection/000_project.png', joint_img)
    # transformed_face_v1[:,2]=0

    # write_numpy_to_obj(gt_joint, filename="head.obj")
    print(1)