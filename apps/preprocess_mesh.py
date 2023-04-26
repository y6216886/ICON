'''
A script to preprocess the obj files into glb files for faster loading.
'''

from pathlib import Path
from tqdm import tqdm
import trimesh
import threadpoolctl

def save_gltf(from_path: Path, target_path: Path):
    m = trimesh.load_mesh(from_path, process=True)
    m.export(target_path, include_normals=True)

def main():
    all_objs = list(sorted(Path('data/thuman2/scans').glob('*/*.obj')))
    target_path = Path('/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2/scans')
    for obj in tqdm(all_objs):
        # print((target_path / obj.name.strip('.obj')/obj.name).with_suffix('.glb'))
        save_gltf(obj, (target_path / obj.name.strip('.obj')/ obj.name).with_suffix('.glb'))

if __name__ == '__main__':
    with threadpoolctl.threadpool_limits(limits=4):
        main()
