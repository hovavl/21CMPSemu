import numpy as np
import pickle
import random
from NN_emulator import emulator
import matplotlib.pyplot as plt

data = pickle.load(
    open('/Users/hovavlazare/GITs/21CMPSemu training data/ps_training_data_mini/samples_09-05-2023_centered_z=7.9.pk',
         'rb'))


def split_data(data, tr_split, val_split, predicted_value):
    training_params = {}
    training_results = []
    val_params = {}
    val_results = []
    testing_params = {}
    testing_results = []

    for param in data[0]['model params'].keys():
        training_params[param] = np.array([])
        val_params[param] = np.array([])
        testing_params[param] = np.array([])
    model_params = list(data[0]['model params'].keys())

    for i, sample in enumerate(data.items()):
        if i < len(data.items()) * tr_split:
            for param in sample[1]['model params'].items():
                training_params[param[0]] = np.append(training_params[param[0]], param[1])
            training_results += [[sample[1][predicted_value]]]
        elif i < len(data.items()) * (tr_split + val_split):
            for param in sample[1]['model params'].items():
                val_params[param[0]] = np.append(val_params[param[0]], param[1])
            val_results += [[sample[1][predicted_value]]]
        else:
            for param in sample[1]['model params'].items():
                testing_params[param[0]] = np.append(testing_params[param[0]], param[1])
            testing_results += [[sample[1][predicted_value]]]

    testing_features = np.array(testing_results)
    features = np.array(training_results)
    val_featurs = np.array(val_results)

    return training_params, features, val_params, val_featurs, testing_params, testing_features, model_params


training_params, features, val_params, val_features, testing_params, testing_features, model_params = \
    split_data(data, 0.85, 0.1, 'xH')

training_params['NU_X_THRESH'] = training_params['NU_X_THRESH'] / 1000
val_params['NU_X_THRESH'] = val_params['NU_X_THRESH'] / 1000
testing_params['NU_X_THRESH'] = testing_params['NU_X_THRESH'] / 1000

# files = [training_params, features, val_params, val_features, testing_params, testing_features, model_params]
# with open('/mini_halos/mini_halos_NN/tau_model_files/training_files.pk.pk.pk', 'wb') as f:
#     pickle.dump(files, f)
#
#
# with open('/mini_halos/mini_halos_NN/tau_model_files/training_files.pk.pk.pk', 'rb') as f:
#     training_params, features, val_params, val_features, testing_params, testing_features, model_params = pickle.load(
#         f)

myEmulator = emulator(training_params, val_params, features, val_features, model_params,
                      hidden_dims=[256, 512, 256], features_band=[0], reg_factor=0.0,
                      dropout_rate=0.0,
                      use_log=False, activation='relu', name='xH_emulator',
                      use_custom_layer=False, use_batchNorm=False
                      )

train_loss, val_loss = myEmulator.train(reduce_lr_factor=0.5, loss_func_name='L2', batch_size=128, verbose=True,
                                        epochs=500, decay_patience_value=10, stop_patience_value=30)
# myEmulator = emulator(restore=True, use_log=False,
#                       files_dir='/Users/hovavlazare/GITs/21CMPSemu/mini_halos/mini_halos_NN/xH_model_files',
#                       name='xH_emulator')
#
#
# train_loss, val_loss = myEmulator.retrain(training_params, features, val_params, val_features, reduce_lr_factor=0.5,
#                                           loss_func_name='L2', batch_size=128, verbose=True,
#                                           epochs=500, decay_patience_value=10, stop_patience_value=30)

# par_arr = myEmulator.dict_to_ordered_arr_np(training_params)
fig = plt.figure()

plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

test_loss, pred = myEmulator.test_l2(testing_params, testing_features)

print(np.median(test_loss))

plt.boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
            medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)
plt.show()
x = 1
myEmulator.save('/Users/hovavlazare/GITs/21CMPSemu/mini_halos/mini_halos_NN/xH_model_files_new')
