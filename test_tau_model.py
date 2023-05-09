import numpy as np
import pickle
import random
from NN_emulator import emulator
import matplotlib.pyplot as plt

with open('/Users/hovavlazare/GITs/21CMPSemu/xH_model_files/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params = pickle.load(
        f)

myEmulator = emulator(restore=True, use_log=False,
                      files_dir='/Users/hovavlazare/GITs/21CMPSemu/xH_model_files',
                      name='xH_emulator')

test_loss, pred = myEmulator.test_l2(testing_params, testing_features)

print(np.median(test_loss))

plt.boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)
plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/emulation_results_xH')

plt.show()