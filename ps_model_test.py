import numpy as np
from matplotlib import pyplot as plt
import pickle
from NN_emulator import emulator
import random
import matplotlib as mpl
import matplotlib.lines as mlines
import seaborn as sns

plt.rc('text', usetex=True)  # render font for tex
plt.rc('font', family='TimesNewRoman')  # use font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 15  # pad is in points...
mpl.rcParams['figure.dpi'] = 500

with open('/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_7-9/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range = pickle.load(
        f)

with open('/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_10-4/training_files.pk', 'rb') as f:
    training_params1, features1, val_params1, val_features1, testing_params1, testing_features1, model_params1, k_range1 = pickle.load(
        f)

myEmulator = emulator(restore=True, use_log=False,
                      files_dir='/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_7-9',
                      name='emulator_7-9_full_range')

myEmulator1 = emulator(restore=True, use_log=False,
                       files_dir='/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_10-4',
                       name='emulator_10-4_full_range')
# print(myEmulator.NN.summary())
# ind = testing_params['L_X'] > 38
# filter_test_params = {}
# for key in testing_params.keys():
#     filter_test_params[key] = testing_params[key][ind]
# filter_test_features = testing_features[ind, :]

filter_test_params = testing_params
filter_test_features = testing_features
print(filter_test_features.shape[0])
test_loss, pred = myEmulator.test_APE(filter_test_params, filter_test_features)

filter_test_params1 = testing_params1
filter_test_features1 = testing_features1
print(filter_test_features1.shape[0])
test_loss1, pred1 = myEmulator1.test_APE(filter_test_params1, filter_test_features1)

# q_16 = np.quantile(test_loss, 0.16)
#
# q_50 = np.quantile(test_loss, 0.50)
# q_84 = np.quantile(test_loss, 0.84)
#
# q_16_val = np.around(q_50 - q_16, 1)
# q_84_val = np.around(q_84 - q_50, 1)
# q_50 = np.around(q_50, 1)
#
# q_16_2 = np.quantile(test_loss1, 0.16)
#
# q_50_2 = np.quantile(test_loss1, 0.50)
# q_84_2 = np.quantile(test_loss1, 0.84)
#
# q_16_val_2 = np.around(q_50_2 - q_16_2, 1)
# q_84_val_2 = np.around(q_84_2 - q_50_2, 1)
# q_50_2 = np.around(q_50_2, 1)
#
# label = f'$z=7.9,  \mathcal{{L}} = {{{q_50}}}_{{{-q_16_val}}}^{{+{q_84_val}}}$'
# label_2 = f'$z=10.4,  \mathcal{{L}} = {{{q_50_2}}}_{{{-q_16_val_2}}}^{{+{q_84_val_2}}}$'
#
# plt.figure(figsize=(12, 8))
#
# dist1 = sns.kdeplot(test_loss, color='navy', linewidth = 3, label = label)
# dist2 = sns.kdeplot(test_loss1, color='salmon', linewidth = 3, label = label_2)
#
# lines_1 = dist1.lines[0]
# x1, y1 = lines_1.get_data()
#
# lines_2= dist2.lines[0]
# x2, y2 = lines_2.get_data()
#
#
#
#
#
#
# ind_16 = x1[np.argmin(np.abs(x1 - q_16))]
# ind_84 = x1[np.argmin(np.abs(x1 - q_84))]
#
# ind_16_2 = x2[np.argmin(np.abs(x2 - q_16_2))]
# ind_84_2 = x2[np.argmin(np.abs(x2 - q_84_2))]
#
# plt.vlines(q_16, 0, 0.3, color='navy', ls='dashed', alpha=1)
# #plt.vlines(q_50, 0, 0.3, color='navy', ls='solid', label=label, alpha=0.5)
# plt.vlines(q_84, 0, 0.3, color='navy', ls='dashed', alpha=1)
#
# plt.vlines(ind_16_2, 0, 0.3, color='salmon', ls='dashed', alpha=1)
# #plt.vlines(q_50_2, 0, 0.3, color='salmon', ls='solid', label=label_2, alpha=0.5)
# plt.vlines(ind_84_2, 0, 0.3, color='salmon', ls='dashed', alpha=1)
#
#
# # print(x2)
# # print('ind: ', ind_16_2, ind_84_2)
#
# plt.fill_between(x1,y1,0, where=np.logical_and(x1 >= ind_16, x1 <= ind_84), color = 'navy', alpha = 0.3)
# plt.fill_between(x2,y2,0, where=np.logical_and(x2 >= ind_16_2, x2 <= ind_84_2), color = 'salmon', alpha = 0.3)
#
# plt.ylim(0, 0.27)
# plt.xlim(0, 25)
# plt.xticks(ticks=[5 * i for i in range(6)], fontsize=28)
# plt.yticks(ticks=[0.05 * i for i in range(6)], fontsize=28)
# plt.ylabel(r'Density', fontsize=30)
# plt.xlabel(r'$\mathcal{L} =  \frac{|y_{real} - y_{pred}|}{y_{real}} \times 100$', fontsize=30)
# plt.legend(frameon=False, fontsize=32)
# plt.tight_layout()
# plt.savefig(f'/Users/hovavlazare/GITs/21CMPSemu/images/errors_plot_no_mini_retrained.png')
#
#
# exit(0)



red_line = mlines.Line2D([], [], color='navy', label=np.round(np.median(test_loss1), decimals=1))
blue_line = mlines.Line2D([], [], color='salmon', label=np.round(np.median(test_loss), decimals=1))
plt.figure(figsize=(10, 8))
box_plot = plt.boxplot([test_loss, test_loss1], whis=(5, 95),
                       whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'deepskyblue'},
                       medianprops={'color': 'navy', 'linewidth': 2}, showfliers=True, labels=['7.9', '10.4'])
for median in box_plot['medians']:
    median.set_color('salmon')
    break

plt.xlabel(r'redshift $z$', fontdict={'fontsize': 26})
plt.ylabel(r'$\frac{|y_{real} - y_{pred}|}{y_{real}} \times 100$', fontdict={'fontsize': 26})
plt.legend(handles=[red_line, blue_line], loc='upper right', frameon=False, prop={"size": 26})
plt.xticks(fontsize=28)
plt.yticks(ticks=[5*(i+1) for i in range(6)],fontsize=26)
plt.ylim(0, 25)
plt.tight_layout()
plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/testing_no_MCGs_boxplot_retrained.png')

exit(0)
x = 2

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 8))

ax[0].boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)

worst_ind = np.argmax(test_loss)

# worst_params = {}
# for key in filter_test_params.keys():
#     worst_params[key] = [filter_test_params[key][worst_ind]]

worst_pred = pred[worst_ind, :]
worst_feature = filter_test_features[worst_ind, :]
print(np.median(test_loss))
# print(worst_params)

# count = 0
# for i in range(pred.shape[0]):
#     sig = pred[i,:]
#     if np.any(sig < 0):
#         count+=1
# print('count: ',count)

ax[1].plot(k_range, worst_feature, c='r', label='Original')
ax[1].plot(k_range, worst_pred, c='b', label='NN reconstructed', linestyle='--')
ax[1].set_xscale('log')
ax[1].set_xlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=20)
ax[1].set_ylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(7.9) + ')', fontsize=15)
ax[1].legend()

plt.show()

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(50, 10))
for i in range(5):
    n = random.randint(0, len(filter_test_params['F_STAR10']) - 1)
    pred_n = pred[n]

    true = filter_test_features[n]
    ax[i].semilogx(k_range, true, 'blue', label='Original')
    ax[i].semilogx(k_range, pred_n, 'red', label='NN reconstructed', linestyle='--')
    ax[i].set_xlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=20)
    ax[i].set_ylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(7.9) + ')', fontsize=15)
    ax[i].legend(fontsize=15)
    ax[i].grid()

plt.show()

x = 1
