import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import statannot 

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{xcolor}', r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans serif' # Choose a nice font here
plt.rcParams['font.size'] = 18

data = pd.read_csv('results_real_data_v1.csv')
data_x = data[data['model'] != 'SDW_MWF']
data_x['impr'] = data['sdr'] + data['snr']
data_x = data_x[data_x['impr'] > 0]

plt.figure(figsize=(20,10))

df = data_x
x = 'model'
y = 'impr'
hue = 'n_ch'
# ax = sns.boxplot(x=x, y=y, data=df, hue=hue, palette="Blues", showfliers=False)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
ax = sns.barplot(x=x, y=y, data=df, hue=hue, palette="Blues")
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
# ax.set_ylim([-15, 20])
# ax = sns.lineplot(x=x, y=y, data=df, hue=hue, palette="Blues")
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

medians = df.groupby(['model'])['sdr'].median()
means = df.groupby(['model'])['sdr'].mean()
std = np.sqrt(np.sqrt(df.groupby(['model'])['sdr'].var()))

# print("MEANS: {}".format(means))
print("STD: {}".format(std))
print("MEDIANS: {}".format(medians))

print('=====')

medians = df.groupby(['model'])['stoi'].median()
means = df.groupby(['model'])['stoi'].mean()
std = np.sqrt(df.groupby(['model'])['stoi'].var())

# print("MEANS: {}".format(means))
print("STD: {}".format(std))
print("MEDIANS: {}".format(medians))


ax.set_xlabel("")
ax.set_ylabel('SDR improvement (dB)')
# ax.legend(title="Input SNR (dB)")

ax.legend(title="Channels", loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fancybox=True)
# snr
# statannot.add_stat_annotation(ax, df, x=x, y=y, hue=hue, boxPairList=[(0, 5), (5, 10)],
#                                 test='Mann-Whitney', textFormat='star', loc='inside', verbose=2)

plt.savefig('compare_nch_sdr_imp.pdf', bbox_inches='tight', transparent=True)