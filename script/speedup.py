import pandas as pd
import numpy as np
import seaborn as sns
import os.path
import matplotlib.pyplot as plt
import getopt
import sys

def func_time_sol(row):
    min=sys.maxsize
    for i in range(0, len(row)):
        x = row[i]
        if x<min and x>0:
           min=x
    return min

def func_index_sol(row):
    min=sys.maxsize
    idx=0
    for i in range(0, len(row)):
        x = row[i]
        if x<min and x>0:
           min=x
           idx=i
    return idx

def func_radix_speedup_sol(row):
    if row['raft_radix_11bits_extra_pass'] ==0:
        return np.NaN
    else:
        return row['time_sol']/row['new_radix_time_sol']

def func_speedup_radix(row):
    if row['raft_radix_11bits_extra_pass'] ==0:
        return np.NaN
    elif row['drtopk_radix']==0:
        return np.NaN
    else:
        return row['drtopk_radix']/row['raft_radix_11bits_extra_pass']

def func_speedup_warp(row):
    min=row['faiss_warp']
    if min>row['faiss_block']:
        min=row['faiss_block']
    if row['grid_select'] ==0:
        return np.NaN
    else:
        return min/row['grid_select']

def calculate(data):
    data_sol=data[['drtopk_bitonic', 'cub', 'sampleselect', 'sampleselect-bucket',
    'sampleselect-quick', 'drtopk_radix','faiss_warp', 'faiss_block']] 
    new_radix_sol=data[['raft_radix_11bits_extra_pass']]  

    data['time_sol']=data_sol.apply(func_time_sol, axis=1)
    data['new_radix_time_sol']=new_radix_sol.apply(func_time_sol, axis=1)    

    data['speedup_radix']=data.apply(func_speedup_radix, axis=1)
    data['speedup_warp']=data.apply(func_speedup_warp, axis=1)
    data['speedup_radix_sol']=data.apply(func_radix_speedup_sol, axis=1)
    return data



if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "k:n:o:")
    k_csv=""
    n_csv=""
    output_csv=""
    for op, value in opts:
        if op == "-k":
            k_csv = value
        elif op == "-n":
            n_csv = value
        elif op == "-o":
            output_csv = value
    
    df = pd.DataFrame()

    k_data = pd.read_csv(k_csv)
    n_data = pd.read_csv(n_csv)

    data = pd.concat([k_data, n_data])
    print(k_data.size, n_data.size, data.size)
    data_res=calculate(data)

    for batch in [1,100]:
        data_batch=data_res.query("batch == @batch")
        for dist in [' Uniform',' Normal',' Unfriendly']:
            dist_name=dist
            if dist[1:9]== "Unfriend":
                dist_name=" Adversarial"

            data_dist=data_batch.query("dist == @dist")

            min_radix=data_dist['speedup_radix'].min()
            max_radix=data_dist['speedup_radix'].max()
            min_warp=data_dist['speedup_warp'].min()
            max_warp=data_dist['speedup_warp'].max()
            min_radix_sol=data_dist['speedup_radix_sol'].min()
            max_radix_sol=data_dist['speedup_radix_sol'].max()

            row={'dist': dist_name, 'batch':batch, 'min_radix':min_radix, 'max_radix':max_radix,
                'min_warp':min_warp, 'max_warp':max_warp,
                'min_radix_sol':min_radix_sol, 'max_radix_sol':max_radix_sol}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            table=str(batch)+'&'+dist_name+'&'+str(round(min_radix,2))+'-'+str(round(max_radix,2))+'&'+str( round(min_warp,2))+'-'+str(round(max_warp,2))+'&'+str(round(min_radix_sol,2))+'-'+str(round(max_radix_sol,2))+'\\\\'
            print(table)
    print(df)
    df.to_csv(output_csv, index=False)