# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,InsetPosition,mark_inset) 
import matplotlib.ticker as ptick
import numpy as np
import pickle

def set_template():                                                               
#    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定          
    plt.rcParams['font.family'] = 'serif' # font familyの設定                     
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定                   
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。         
    plt.rcParams['axes.labelsize'] = 25 # ラベルのフォントサイズ                  
    plt.rcParams['xtick.labelsize'] = 25 # x軸メモリのフォントサイズ              
    plt.rcParams['ytick.labelsize'] = 25 # y軸メモリのフォントサイズ              
    plt.rcParams['xtick.direction'] = 'in' # x axis in                            
    plt.rcParams['ytick.direction'] = 'in' # y axis in                            
    plt.rcParams["xtick.minor.visible"] = False #x軸補助目盛りの追加              
    plt.rcParams["ytick.minor.visible"] = True #y軸補助目盛りの追加               
    plt.rcParams['axes.linewidth'] = 0.8 # axis line width                        
    plt.rcParams['axes.grid'] = True # make grid (True or False)                  
    plt.rcParams['axes.axisbelow'] = True                                         
    plt.rcParams['legend.fontsize'] = 20 #                                        
    plt.rcParams["legend.fancybox"] = False # 丸角                                
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし       
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更                   
    plt.rcParams["legend.handlelength"] = 0.8 # 凡例の線の長さを調節              
    plt.rcParams["legend.labelspacing"] = 0.7 # 垂直（縦）方向の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 0.5 # 凡例の線と文字の距離の長さ       
    plt.rcParams["legend.markerscale"] = 1.0 # 点がある場合のmarker scale

if __name__ == "__main__":
    set_template()
    nst = 95100
    nls = 115000
    nin = 10
    allsteps = np.arange(nst,nls,nin)
    # data load
    f = open("./recerrors.pickle","rb")
    errors_seq = pickle.load(f)
    # steps  = allsteps[:-1]

    # errors
    nepochs = len(errors_seq)
    errors=[]
    for batches in errors_seq:
        steps = batches.shape[0]
        for step in range(steps):
            errors.append(batches[step])

    # make postdata
    mean_errors = []
    stds        = []
    for error in errors:
        mean_errors.append(np.mean(error))
        stds.append(np.std(error))

    mean_errors = np.array(mean_errors)
    stds        = np.array(stds)

    steps = np.arange(1,len(errors)+1)

    # display reconstruction
    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111)

    ax.plot(steps,mean_errors,marker='.')
    ax.fill_between(steps, mean_errors + stds, mean_errors - stds, alpha=0.2)

    ax.set_xlabel('step')
    ax.set_ylabel(r'Average of Relative error $|\mathbf{Y}_{\mathrm{rec}}-\mathbf{Y}_{\mathrm{org}}|_{1} / |\mathbf{Y}_{\mathrm{org}}|_{1}$')

    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(3,-4))

    # plt.show()
    plt.savefig('result_relativeerors.pdf')


