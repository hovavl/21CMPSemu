import numpy as np
import pickle
import random
from NN_emulator import emulator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import seaborn as sns
from scipy.interpolate import splev, splrep


plt.rc('text', usetex=True)  # render font for tex
plt.rc('font', family='TimesNewRoman')  # use font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 15  # pad is in points...
mpl.rcParams['figure.dpi'] = 500



with open('/Users/hovavlazare/GITs/21CMPSemu/NN/tau_model_files/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params = pickle.load(
        f)


with open('/Users/hovavlazare/GITs/21CMPSemu/NN/xH_model_files/training_files.pk', 'rb') as f:
    training_params_xH, features_xH, val_params_xH, val_features_xH, testing_params_xH, testing_features_xH, model_params_xH = pickle.load(
        f)

Emulator_tau = emulator(restore=True, use_log=False,
                       files_dir='/Users/hovavlazare/GITs/21CMPSemu/NN/tau_model_files',
                       name='tau_emulator')

Emulator_xH = emulator(restore=True, use_log=False,
                       files_dir='/Users/hovavlazare/GITs/21CMPSemu/NN/xH_model_files',
                       name='xH_emulator')


#ind = np.logical_and(np.logical_or(testing_params['L_X'] < 38.5, testing_params['L_X'] > 41.5), testing_params['L_X'] >= 38)
# ind = testing_params['L_X'] > 41.5
# filter_test_params = {}
# for key in testing_params.keys():
#     filter_test_params[key] = testing_params[key][ind]
# filter_test_features = testing_features[ind, :]

test_loss, pred_tau = Emulator_tau.test_l2(testing_params, testing_features)
test_loss1, pred = Emulator_xH.test_l2(testing_params_xH, testing_features_xH)



q_16 = np.quantile(test_loss, 0.16)

q_50 = np.quantile(test_loss, 0.50)
q_84 = np.quantile(test_loss, 0.84)

q_16_val = np.around(q_50 - q_16, 4)
q_84_val = np.around(q_84 - q_50, 4)
q_50 = np.around(q_50, 4)

q_16_2 = np.quantile(test_loss1, 0.16)

q_50_2 = np.quantile(test_loss1, 0.50)
q_84_2 = np.quantile(test_loss1, 0.84)

q_16_val_2 = np.around(q_50_2 - q_16_2, 3)
q_84_val_2 = np.around(q_84_2 - q_50_2, 3)
q_50_2 = np.around(q_50_2, 3)

label = f'$\\tau,  \mathcal{{L}} = {{{q_50*10000}}}_{{{-q_16_val*10000}}}^{{+{q_84_val*10000}}} {{\cdot 10^{{-4}}}}$'
label_2 = f'$x_{{HI}},  \mathcal{{L}} = {{{q_50_2*1000}}}_{{{-q_16_val_2*1000}}}^{{+{q_84_val_2*1000}}} {{\cdot 10^{{-3}}}}$'

plt.figure(figsize=(12, 8))
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

dist1 = sns.kdeplot(test_loss, color='navy', linewidth = 3, label = label)
# dist2 = sns.kdeplot(test_loss1, color='navy', linewidth = 3, label = label_2)

lines_1 = dist1.lines[0]
x1, y1 = lines_1.get_data()

# lines_2= dist2.lines[0]
# x2, y2 = lines_2.get_data()



plt.vlines(q_16, 0, 4000, color='navy', ls='dashed', alpha=1)
#plt.vlines(q_50, 0, 0.3, color='navy', ls='solid', label=label, alpha=0.5)
plt.vlines(q_84, 0, 4000, color='navy', ls='dashed', alpha=1)

# plt.vlines(q_16_2, 0, 125, color='navy', ls='dashed', alpha=1)
# #plt.vlines(q_50_2, 0, 0.3, color='salmon', ls='solid', label=label_2, alpha=0.5)
# plt.vlines(q_84_2, 0, 125, color='navy', ls='dashed', alpha=1)




# ind_16 = x2[np.argmin(np.abs(x2- q_16_2))]
# ind_84 = x2[np.argmin(np.abs(x2- q_84_2))]

ind_16 = x1[np.argmin(np.abs(x1- q_16))]
ind_84 = x1[np.argmin(np.abs(x1- q_84))]

plt.fill_between(x1,y1,0, where=np.logical_and(x1 >= ind_16, x1 <= ind_84), color = 'navy', alpha = 0.3)
# plt.fill_between(x2,y2,0, where=np.logical_and(x2 >= ind_16, x2 <= ind_84), color = 'navy', alpha = 0.3)

plt.ylim(0, 4000)
plt.xlim(0, 0.0012)
plt.xticks(ticks=[0.0004*i for i in range(4)],fontsize=28)
plt.yticks(fontsize=28)
plt.ylabel(r'Density', fontsize=30)
plt.xlabel(r'$\mathcal{L} = |y_{true} - y_{pred}| $', fontsize=30)

# ax[1].set_ylabel(r'Density', fontsize=30)
# ax[0].xlabel(r'$\mathcal{L} = |y_{true} - y_{pred}| $', fontsize=30)

plt.legend(frameon=False, fontsize=40)
plt.tight_layout()
plt.savefig(f'/Users/hovavlazare/GITs/21CMPSemu/images/errors_tau.png')
exit(0)






print(np.median(test_loss))
x=1

red_line = mlines.Line2D([], [], color='red', label=np.round(np.median(test_loss_tau),decimals =4))
blue_line = mlines.Line2D([], [], color='blue', label=np.round(np.median(test_loss), decimals=4))
plt.figure(figsize=(8,8))
box_plot = plt.boxplot([test_loss_tau,test_loss], whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5},labels=[r'$\tau$', r'$x_{HI}$'], showfliers=True)
for median in box_plot['medians']:
    median.set_color('b')
    break

plt.yticks(fontsize=24)
# plt.xlabel('', fontsize = 28)
plt.legend(handles=[red_line, blue_line], loc='upper right', frameon=False, prop={"size":26})
plt.ylabel(r'$|y_{true} - y_{pred}|$',fontsize = 28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.tight_layout()
plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/results_emulators_xH-tau.png')

plt.show()