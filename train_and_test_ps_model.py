import numpy as np
import pickle
import random
from NN_emulator import emulator
import matplotlib.pyplot as plt
from Classifier import SignalClassifier

data = pickle.load(
   open('/Users/hovavlazare/GITs/21CMPSemu training data/ps_training_data/centered_samples_10-4_2023-04-14.pk', 'rb'))


# extra_data = pickle.load(
#     open('/Users/hovavlazare/PycharmProjects/global_signal_model/ps_data/extra_data_spline.pk', 'rb'))


def classify_signal(signal):
    minimum = np.min(signal)
    maximum = np.max(signal)
    if minimum != signal[0] and maximum - minimum > 1 and np.all(signal > 0.5):
        return 5
    elif maximum - minimum < 1 and np.all(signal > 0.5):
        return 0
    elif maximum - minimum < 10 and np.all(signal > 0.5):
        return 1
    elif maximum - minimum < 100 and np.all(signal > 0.5):
        return 2
    elif maximum - minimum < 1000 and np.all(signal > 0.5):
        return 3
    elif maximum - minimum < 10000 and np.all(signal > 0.5):
        return 4
    return -1


def organize_data(data):
    params = []
    class_0_params = []
    powerspectra = []
    class_counter = {}
    for i in range(0, 6):
        class_counter[i] = 0

    model_params = list(data[0]['model params'].keys())
    # model_params += ['tau', 'x_HI']
    my_items = list(data.items())
    # random.shuffle(my_items)
    counter = 0
    for i, sample in enumerate(my_items):
        counter += 1
        if i <= len(my_items):
            reduced_sample = sample[1]['ps'][30:89]
            class_num = classify_signal(reduced_sample)

            if (0 < class_num < 6 and class_counter[class_num] < 2500): #or (class_num == 5 and class_counter[5] < 7000):
                class_counter[class_num] += 1
                params += [list(sample[1]['model params'].values())]
                powerspectra += [reduced_sample]
            elif class_num == 0 and class_counter[0] < 1000:
                class_0_params += [list(sample[1]['model params'].values())]
                class_counter[0] += 1
            if np.all(np.array(list(class_counter.values()))[1:5] > 2500):
                break
    powerspectra = np.array(powerspectra)
    params = np.array(params)
    k_range = data[0]['k'][30:89]
    return powerspectra, params, model_params, k_range, class_0_params


myClassifier = SignalClassifier(restore=True,
                                files_dir='/Users/hovavlazare/GITs/21CMPSemu/classifier_files',
                                name='classify_NN')


def classify_test_data(test_params_dict, test_features):
    classes = np.squeeze(np.round(myClassifier.predict(test_params_dict)).astype(bool))
    test_params_arr = np.array([test_params_dict[k] for k in test_params_dict.keys()]).T
    filter_test_params_arr = test_params_arr[classes]
    filter_test_params_dict = dict(zip(list(test_params_dict.keys()), filter_test_params_arr))
    filter_features = test_features[classes]
    return filter_test_params_dict, filter_features





def divide_data(params, powerspectra, model_params, tr_split, val_split):
    all_data = list(zip(params, powerspectra))
    random.shuffle(all_data)
    s_params, s_powerspectra = zip(*all_data)
    s_params = np.array(s_params)
    s_powerspectra = np.array(s_powerspectra)
    tr_params = s_params[:int(s_params.shape[0] * tr_split), :]
    tr_powerspectra = s_powerspectra[:int(s_powerspectra.shape[0] * tr_split), :]
    val_params = s_params[int(s_params.shape[0] * tr_split) + 1:int(s_params.shape[0] * (tr_split + val_split)), :]
    val_powerspectra = s_powerspectra[int(s_powerspectra.shape[0] * tr_split) + 1:int(
        s_powerspectra.shape[0] * (tr_split + val_split)), :]
    test_params = s_params[int(s_params.shape[0] * (tr_split + val_split)) + 1:, :]
    test_powerspectra = s_powerspectra[int(s_powerspectra.shape[0] * (tr_split + val_split)) + 1:, :]

    tr_par_dict = {}
    val_par_dict = {}
    test_par_dict = {}
    for j, par_name in enumerate(model_params):
        tr_par_dict[par_name] = tr_params[:, j]
        val_par_dict[par_name] = val_params[:, j]
        test_par_dict[par_name] = test_params[:, j]
    return tr_par_dict, tr_powerspectra, val_par_dict, val_powerspectra, test_par_dict, test_powerspectra


powerspectra, params, model_params, k_range, class_0_params = organize_data(data)

training_params, features, val_params, val_features, testing_params, testing_features = \
    divide_data(params, powerspectra, model_params, 0.80, 0.10)

# training_params, features, val_params, val_features, testing_params, testing_features, k_range, model_params = \
#     split_data(0.85, 0.1,
#                data, extra_data)
training_params['NU_X_THRESH'] = training_params['NU_X_THRESH'] / 1000
val_params['NU_X_THRESH'] = val_params['NU_X_THRESH'] / 1000
testing_params['NU_X_THRESH'] = testing_params['NU_X_THRESH'] / 1000
f_test_params, f_test_features = classify_test_data(testing_params, testing_features)


files = [training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range]
with open('/Users/hovavlazare/GITs/21CMPSemu/centered_model_files_10-4/centered_training_files.pk', 'wb') as f:
    pickle.dump(files, f)

exit(0)
with open('/Users/hovavlazare/GITs/21CMPSemu/model_files_10-4/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range = pickle.load(
        f)


# tr_splits = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
# loss_means = []
# loss_stds = []
# for tr_split in tr_splits:
#     training_params, features, val_params, val_features, testing_params, testing_features, k_range, model_params = \
#         split_data(tr_split, 0.1,
#                    data, extra_data)
#     training_params['NU_X_THRESH'] = training_params['NU_X_THRESH'] / 1000
#     val_params['NU_X_THRESH'] = val_params['NU_X_THRESH'] / 1000
#     testing_params['NU_X_THRESH'] = testing_params['NU_X_THRESH'] / 1000
#     myEmulator = emulator(training_params, val_params, features, val_features, model_params,
#                           hidden_dims=[288, 512, 512, 288], features_band=k_range, reg_factor=0.0, dropout_rate=0.0,
#                           use_log=True, activation='linear', name='small_diff_emulator'
#                           )
#     train_loss, val_loss = myEmulator.train(reduce_lr_factor=0.5, loss_func_name='APE', batch_size=1024, verbose=True,
#                                             epochs=500, decay_patience_value=10, stop_patience_value=30)
#
#
#     ind = testing_params['L_X'] > 38
#     filter_test_params = {}
#     for key in testing_params.keys():
#         filter_test_params[key] = testing_params[key][ind]
#     filter_test_features = testing_features[ind, :]
#
#     test_loss, pred = myEmulator.test_APE(filter_test_params, filter_test_features)
#     loss_means += [np.mean(test_loss)]
#     loss_stds += [np.std(test_loss)]
#
# plt.errorbar(tr_splits, loss_means, yerr=loss_stds, capsize=4)
# plt.savefig('loss with data amount')
# plt.show()


myEmulator = emulator(training_params, val_params, features, val_features, model_params,
                      hidden_dims=[288, 512, 512, 288, 512, 1024], features_band=k_range, reg_factor=0.0,
                      dropout_rate=0.0,
                      use_log=False, activation='linear', name='emulator_10-4',
                      )

print(myEmulator.NN.summary())
train_loss, val_loss = myEmulator.train(reduce_lr_factor=0.5, loss_func_name='APE', batch_size=128, verbose=True,
                                        epochs=500, decay_patience_value=10, stop_patience_value=30)

par_arr = myEmulator.dict_to_ordered_arr_np(training_params)
fig = plt.figure()

plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

ind = testing_params['L_X'] > 38
filter_test_params = {}
for key in testing_params.keys():
    filter_test_params[key] = testing_params[key][ind]
filter_test_features = testing_features[ind, :]

test_loss, pred = myEmulator.test_MRE(filter_test_params, filter_test_features)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 8))

ax[0].boxplot(test_loss, whis=(5, 95), whiskerprops={'ls': 'dotted', 'linewidth': 1, 'color': 'b'},
              medianprops={'color': 'r', 'linewidth': 0.5}, showfliers=True)
worst_ind = np.argmax(test_loss)

worst_params = {}
for key in testing_params.keys():
    worst_params[key] = [filter_test_params[key][worst_ind]]

worst_pred = pred[worst_ind, :]
worst_feature = filter_test_features[worst_ind, :]
print('worst feature: ', worst_feature)
print(np.median(test_loss))
print(worst_params)

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

ig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
for i in range(3):
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

myEmulator.save('/Users/hovavlazare/GITs/21CMPSemu/model_files_10-4')
x = 1
