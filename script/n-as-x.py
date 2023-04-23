import pandas as pd
import numpy as np
import seaborn as sns
import os.path
import matplotlib.pyplot as plt
import getopt
import sys

def genX(df, name):
    x=np.asarray(df[name].to_numpy().nonzero()).transpose()
    return x

def nonzero(df,name):
    df2 = df[df[name] != 0]
    return df2[name]


def genFig(df,ax,title):
    num_rows=len(df.index)
    x=np.arange(num_rows)

    ax.plot(genX(df,"sampleselect"),nonzero(df,"sampleselect"), 'o-',label="SampleSelect")
    ax.plot(genX(df,"sampleselect-bucket"),nonzero(df,"sampleselect-bucket"),'+-',label="BucketSelect")
    ax.plot(genX(df,"sampleselect-quick"),nonzero(df,"sampleselect-quick"),'*-',label="QuickSelect")   
    ax.plot(genX(df,"cub"),nonzero(df,"cub"),'x-', label="Sort")
    ax.plot(genX(df,"drtopk_radix"),nonzero(df,"drtopk_radix"),'p-', label="RadixSelect")
    ax.plot(genX(df,"drtopk_bitonic"),nonzero(df,"drtopk_bitonic"),'h-',label="Bitonic Top-K") 

    ax.plot(genX(df,"faiss_warp"), nonzero(df,"faiss_warp"),'4-',label="WarpSelect")  
    ax.plot(genX(df,"faiss_block"),nonzero(df,"faiss_block"),'^-',label="BlockSelect")   
    ax.plot(genX(df,"raft_radix_11bits_extra_pass"), nonzero(df,"raft_radix_11bits_extra_pass"),'8-',color="olivedrab",label="AIR Top-K")  
    ax.plot(genX(df,"grid_select"),nonzero(df,"grid_select"),'2-',label="GridSelect")  

    plt.sca(ax)
    plt.xticks(x,df["n_power"].to_numpy().astype(int),rotation = 'vertical')       
    plt.yscale('log')
    plt.title(title)
    handles,labels = ax.get_legend_handles_labels()
    return handles,labels


def drawFig(input_csv, output_png):
    csv=input_csv
    png_all_name=output_png
    print(csv, png_all_name)

    data= pd.read_csv(csv)
    count=0
    fig, axes = plt.subplots(nrows=3, ncols=6,figsize=(16,12))
    fig.text(0, 0.5, "Running Time (unit:ms)", va='center', rotation='vertical',fontsize=14)
    fig.text(0.55, 0.02, r'$log_{2}(N)$', ha='center',fontsize=14)

    for dist in [" Uniform", " Normal", " Unfriendly"]:
        data_dist=data.query("dist == @dist")

        for bs in [1,100]: 
            data_bs=data_dist.query("batch == @bs")

            for k_power in [5,8,15]:
                k=pow(2,k_power)
                data_k=data_bs.query("k == @k")
                print(dist, bs, k, data_k.size)
                #print(data_k)
                if data_k.size!=0:  
                    if dist[1:9] != "Unfriend":
                        title = "("+str(count)+") Batch="+str(bs)+" K="+(r'$2^{%s}$' %(str(k_power)))+" "+dist[1:9] #(r'$2^{%s}$' %(str(n_power)))
                    else:      
                        dist_name="Radix-adversarial"            
                        title = "("+str(count)+") Batch="+str(bs)+" K="+(r'$2^{%s}$' %(str(k_power)))+" \n "+dist_name #(r'$2^{%s}$' %(str(n_power)))
                      
                    data_k=data_k.query("raft_radix_11bits_extra_pass != 0")
                    if(count==0):
                       handles,labels =genFig(data_k,axes.flat[count],title)
                    else:
                       genFig(data_k,axes.flat[count],title)
                    count=count+1

    leg=fig.legend(handles, labels,ncol=5,bbox_to_anchor=(0.8,1),fontsize='large')
    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.07, wspace=0.15, hspace=0.27)

    plt.savefig(png_all_name)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
    input_csv=""
    output_png=""
    for op, value in opts:
        if op == "-i":
            input_csv = value
        elif op == "-o":
            output_png = value
    drawFig(input_csv, output_png)