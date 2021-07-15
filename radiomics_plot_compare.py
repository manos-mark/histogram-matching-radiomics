#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:15:41 2021

@author: george
"""

import matplotlib.pyplot as plt
import pandas as pd 



dataset = ['Normal' ,'Pipeline_1', 'Pipeline_2', 'average_hist' ,'p2_made_image','Pieline_2_george']

df =  pd.read_excel('Radiomic_features_updated.xlsx',sheet_name=dataset[0])

df2 =  pd.read_excel('Radiomic_features_updated.xlsx',sheet_name=dataset[3])

names = [col for col in df.columns][2:11]

nam = [ 'Entropy','	Homogeneity','Contrast']
r = 0

for j in range(0,len(names)):
    
    data = [ [float(i) for i in df[names[j]][1:11]],[float(i) for i in df[names[j]][12:22]],    [float(i) for i in df[names[j]][23:33]],
             [float(i) for i in  df2[names[j]][1:11]],[float(i) for i in df2[names[j]][12:22]] ,[float(i) for i in df2[names[j]][23:33]]]
    
    fig = plt.figure(figsize =(10, 7))
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    if names[j].find("U")==-1: 
        channel = names[j]
    
    fig.suptitle(channel+'_'+nam[r%3], fontsize=14, fontweight='bold')
    
    ax.set_xticklabels(['CS before  ', 'DU before ','FG before','CS after ','DU after ' ,'FG after'])
    bp = ax.boxplot(data)
    plt.savefig(channel+'_'+nam[r%3]+'made.png', dpi=300, bbox_inches='tight')
    r+=1
    plt.show()
