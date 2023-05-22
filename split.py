import numpy as np
def get_subject_list():
        dataset="/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2"
        subject_list = []


        full_txt = '/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/data/thuman2/all.txt'
        print(f"split {full_txt} into train/val/test")

        full_lst = np.loadtxt(full_txt, dtype=str)
        full_lst = [dataset + "/" + item for item in full_lst]
        [train_lst, test_lst, val_lst] = np.split(full_lst, [
            500,
            500 + 5,
        ])

        np.savetxt(full_txt.replace("all", "train"), train_lst, fmt="%s")
        np.savetxt(full_txt.replace("all", "test"), test_lst, fmt="%s")
        np.savetxt(full_txt.replace("all", "val"), val_lst, fmt="%s")

        print(f"load from {split_txt}")
        subject_list += np.loadtxt(split_txt, dtype=str).tolist()



        # subject_list = ["thuman2/0008"]
        return subject_list

if __name__=='__main__':
  get_subject_list()