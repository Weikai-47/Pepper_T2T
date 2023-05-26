import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np

data = pd.read_excel('./副本Andean_coverage6.xlsx')
data['ONT_position'] = 0
data.columns = ['HiFi_chr','HiFi_start','HiFi_end','HiFi_value','HiFi_position',
                'NGS_chr','NGS_start','NGS_end','NGS_value','NGS_position',
                'ONT_chr','ONT_start','ONT_end','ONT_value','ONT_position']
list_data = list(data.groupby('NGS_chr'))

chr_dict = {}
for i in list_data:
    chr_dict[i[0]] = i[1]

labels = [0]
count = 0
for i in range(1,14):
    for j in ['HiFi_start','NGS_start','ONT_start','HiFi_end','NGS_end','ONT_end']:
        chr_dict['chr{}'.format(i)][j] += count
    count = chr_dict['chr{}'.format(i)].iloc[-1,:]['NGS_end']
    labels.append(count/1000000)

data1 = chr_dict['chr1']
for i in range(2,14):
    data1 = pd.concat([data1,chr_dict['chr{}'.format(i)]])

for i in ['NGS','ONT','HiFi']:
    data1[i+'_position'] = (data1[i+'_start'].astype('float') + data1[i+'_end'].astype('float'))/2.0


name = ['HiFi','NGS','ONT']
color = ['pink','aqua','lavender']
for i in range(0,3):
    plt.subplot(3, 1, i+1)
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    print(data1[name[i]+'_value'])
    plt.bar(data1[name[i]+'_position'] / 1000000, data1[name[i]+'_value'], width=1.0, color=color[i])
    plt.scatter(data1[name[i]+'_position'] / 1000000, data1[name[i]+'_value'], color='black', s=0.8, alpha=0.6, marker='s')
    plt.ylabel(name[i]+' Sequence Depth', fontsize=14)
    plt.xticks(labels)
plt.show()
