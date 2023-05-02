import numpy as np
from matplotlib import pyplot as plt
import pickle
from NN_emulator import emulator
import random

filename = '/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_10-4/centered_training_files'

with open(filename, 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range = pickle.load(
        f)

myEmulator = emulator(restore=True, use_log=False,
                      files_dir='/Users/hovavlazare/GITs/21CMPSemu/experimental/model_files_10-4',
                      name='emulator_10-4_full_range')

for i, layer in enumerate(myEmulator.NN.layers):

    if (i + 1) % 3 == 0:
        layer.trainable = False
    else:
        layer.trainable = True

print(myEmulator.NN.summary())

train_loss, val_loss = myEmulator.retrain(x_tr=training_params,
                                          y_tr=features,
                                          x_val=val_params,
                                          y_val=val_features,
                                          reduce_lr_factor=0.5, loss_func_name='APE', batch_size=128, verbose=True,
                                          epochs=500, decay_patience_value=10, stop_patience_value=30)

fig = plt.figure()

plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()


test_loss, pred = myEmulator.test_APE(testing_params, testing_features)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 8))

ax[0].boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)
worst_ind = np.argmax(test_loss)

# worst_params = {}
# for key in filter_test_params.keys():
#     worst_params[key] = [filter_test_params[key][worst_ind]]

worst_pred = pred[worst_ind, :]
worst_feature = testing_features[worst_ind, :]
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
    n = random.randint(0, len(testing_params['F_STAR10']) - 1)
    pred_n = pred[n]

    true = testing_features[n]
    ax[i].semilogx(k_range, true, 'blue', label='Original')
    ax[i].semilogx(k_range, pred_n, 'red', label='NN reconstructed', linestyle='--')
    ax[i].set_xlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=20)
    ax[i].set_ylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(7.9) + ')', fontsize=15)
    ax[i].legend(fontsize=15)
    ax[i].grid()

plt.show()

myEmulator.save('/Users/hovavlazare/GITs/21CMPSemu/experimental/centered_model_files_10-4')
