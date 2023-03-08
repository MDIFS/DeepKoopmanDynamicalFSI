# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,InsetPosition,mark_inset) 
import numpy as np

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
    # data load
    losses = np.load('./losses.npy')
    epochs = np.arange(len(losses))
    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(111)

    # display reconstruction
    ax.plot(epochs,losses,marker='.')

    #
    ax2 =  plt.axes([0,0,1,1])
    ip = InsetPosition(ax, [0.15,0.2,0.82,0.5])
    ax2.set_axes_locator(ip)
    ax2.set_xlim(epochs[-800],epochs[-1])
    ax2.set_ylim(0.0,10.0)
    ax2.tick_params(labelbottom=False,labelleft=True,labelright=False,labeltop=False)
    mark_inset(ax,ax2,loc1=4,loc2=3,fc='None',ec='black',linestyle='--')
    ax2.plot(epochs[-800:-1],losses[-800:-1])
    #

    ax.set_xlabel('epoch')
    ax.set_ylabel(r'Loss $|X_{train}-X_{prd}|^2_{Fro}$')

    # plt.show()
    plt.savefig('result_losses.pdf')

    print(losses[-1])
