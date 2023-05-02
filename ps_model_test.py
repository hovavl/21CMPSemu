import numpy as np
from matplotlib import pyplot as plt
import pickle
from NN_emulator import emulator
import random
import matplotlib as mpl
import matplotlib.lines as mlines


plt.rc('text', usetex=True)  # render font for tex
plt.rc('font', family='TimesNewRoman')  # use font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
#plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 15  # pad is in points...
mpl.rcParams['figure.dpi'] = 300


with open('/Users/hovavlazare/GITs/21CMPSemu/mini_halos/mini_halos_NN/model_files_7-9/training_files', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range = pickle.load(
        f)

myEmulator = emulator(restore=True, use_log=False,
                      files_dir='/Users/hovavlazare/GITs/21CMPSemu/mini_halos/mini_halos_NN/model_files_7-9',
                      name='emulator_7-9_mini')

print(myEmulator.NN.summary())


# ind = testing_params['L_X'] > 38
# filter_test_params = {}
# for key in testing_params.keys():
#     filter_test_params[key] = testing_params[key][ind]
# filter_test_features = testing_features[ind, :]

filter_test_params = testing_params
filter_test_features = testing_features
print(filter_test_features.shape[0])
test_loss, pred = myEmulator.test_APE(filter_test_params, filter_test_features)

red_line = mlines.Line2D([], [], color='red', label='median')

plt.boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)
plt.title('Testing set statistics z = 7.9')
plt.ylabel(r'$\frac{|y_{real} - y_{pred}|}{y_{real}}$', fontdict={'fontsize': 16})
plt.legend(handles=[red_line], loc='upper right', frameon=False, prop={"size": 16})
#plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/results_1')

plt.show()





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


x=1
