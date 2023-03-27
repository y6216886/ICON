import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
# read the existing Excel file into a DataFrame
# df = pd.read_csv("/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/test_excel.csv")
# root="/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline"
# pathlist=list(glob.glob(root+"/*/"+"/*/"+"test_results.npy"))
# for path in pathlist: 
#   print(path)
#   results= pd.Series(np.load(path,allow_pickle=True))
#   print(results)
#   # results=np.load(path)
#   print(path)
#   df = df.append(results, ignore_index=True)

# # write the updated DataFrame back to the Excel file
# df.to_csv("/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/test_excel.csv", index=False)


import csv
root="/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/baseline"
pathlist=list(glob.glob(root+"/*/"+"/*/"+"test_results.npy"))
# d1 = {'name':'jon', 'age':4}
# d2 = {'name':'joe', 'age':34, 'height':100}

csv_columns = ['exp_name','cape-easy-chamfer', 'cape-easy-p2s', 'cape-easy-NC', 'cape-hard-chamfer', 'cape-hard-p2s', 'cape-hard-NC','thuman2-NC', 'thuman2-chamfer', 'thuman2-p2s']


with open('/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/results/file_won.csv', 'a') as f:
      for i, path in tqdm(enumerate(pathlist)): 
        expname=path.split("/")[-3]
        results= pd.Series(np.load(path,allow_pickle=True))
        keylist=list(results[0].item().keys())
        values_list=list(results[0].item().values())
        if i==0:  
            wr = csv.DictWriter(f, fieldnames=csv_columns)
            # wr.writeheader()
        results_modi={'exp_name': expname}
        for keys_, values_ in zip(keylist,values_list):
          results_modi[keys_]='{:.5f}'.format( values_.item())[:-1]   # '{:.5f}'.format(math.sqrt(float('3')))[:-1]
        wr.writerow(results_modi)
