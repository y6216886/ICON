#import os
# import torch
# from tqdm import tqdm
# import time


# def check_mem(cuda_device):
#     devices_info = os.popen(
#         '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
#         "\n")
#     print(devices_info)
#     cuda_device=int(cuda_device)
#     total, used = devices_info[cuda_device].split(',')
#     return total, used


# def occumpy_mem(cuda_device):
#     total, used = check_mem(cuda_device)
#     total = int(total)
#     used = int(used)
#     max_mem = int(total * 0.9)
#     block_mem = max_mem - used
#     # x = torch.cuda.FloatTensor(256, 1024, block_mem)
#     x = torch.FloatTensor(256, 1024, block_mem).cuda(cuda_device)
#     del x


# if __name__ == '__main__':
#     import argparse
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device_ids', help='device_ids', type=str, required=True)
#     parser.add_argument('--time', help='occumpy time(s)', type=int, default=1000000)
#     args = parser.parse_args()
#     for cuda_device in args.device_ids:
#         occumpy_mem(cuda_device)
#     for _ in tqdm(range(args.time)):
#         time.sleep(1)
#     print('Done')

import os
import torch
from tqdm import tqdm
import time


# def check_mem(cuda_device):
#     devices_info = os.popen(
#         '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
#         "\n")
#     total, used = devices_info[cuda_device].split(',')
#     return total, used


# def occumpy_mem(cuda_device,index):
#     print(cuda_device,index)
#     total, used = check_mem(cuda_device)
#     total = int(total)
#     used = int(used)
#     max_mem = int(total*0.8 )
    
#     block_mem = max_mem - used
#     print(block_mem)
#     x = torch.FloatTensor(256, 1024, block_mem).cuda(index)
#     del x

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device, occ):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    
    max_mem = int(total )
    
    block_mem = int((total - used)*occ)
    block_mem=max(block_mem,0)
    print(total, used,max_mem, block_mem,"total, used, max_mem, block_mem")
    x = torch.FloatTensor(256,1024,block_mem).to(torch.device(f"cuda:{cuda_device}"))
    del x

if __name__ == '__main__':
    import argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device_ids', help='device_ids', type=int, nargs="+",
                        default=list(range(torch.cuda.device_count())))
    parser.add_argument('--occ', help='occumpy memory', type=float, default=0.5)
    parser.add_argument('--time', help='occumpy time(s)', type=int, default=1000000)
    args = parser.parse_args()
    print(args.device_ids)
    for cuda_device in args.device_ids:
        occumpy_mem(cuda_device, args.occ)
    for _ in tqdm(range(args.time)):
        time.sleep(1)
    print('Done')

    ##python /mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/utils/train_nerf.py --device_ids 0 1 2 3