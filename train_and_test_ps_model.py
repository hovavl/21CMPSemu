import numpy as np
import pickle
import random
from NN_emulator import emulator, CustomLayer
import matplotlib.pyplot as plt
import h5py

data = pickle.load(
    open('/Users/hovavlazare/PycharmProjects/global_signal_model/ps_data/bigger_data_set_2023-03-12.pk', 'rb'))
extra_data = pickle.load(
    open('/Users/hovavlazare/PycharmProjects/global_signal_model/ps_data/extra_data_spline.pk', 'rb'))


def classify_signal(signal):
    minimum = np.min(signal)
    maximum = np.max(signal)
    if minimum != signal[0] and np.all(signal > 0.5):
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
    powerspectra = []
    class_counter = {}
    for i in range(1,6):
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

            if class_num > 0 and class_counter[class_num] < 3500:
                class_counter[class_num] += 1
                params += [list(sample[1]['model params'].values())]
                powerspectra += [reduced_sample]
            if np.all(np.array(list(class_counter.values())) > 3400):
                break
    powerspectra = np.array(powerspectra)
    params = np.array(params)
    k_range = data[0]['k'][30:89]
    return powerspectra, params, model_params, k_range


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

    x = 1
    tr_par_dict = {}
    val_par_dict = {}
    test_par_dict = {}
    for j, par_name in enumerate(model_params):
        tr_par_dict[par_name] = tr_params[:, j]
        val_par_dict[par_name] = val_params[:, j]
        test_par_dict[par_name] = test_params[:, j]
    return tr_par_dict, tr_powerspectra, val_par_dict, val_powerspectra, test_par_dict, test_powerspectra





def split_data(training_split, val_split, data, extra_data):
    training_params = {}
    training_powerspectra = []
    testing_params = {}
    testing_powerspectra = []
    val_params = {}
    val_powerspectra = []
    model_params = list(data[0]['model params'].keys())
    # model_params += ['tau', 'x_HI']
    for param in model_params:
        training_params[param] = np.array([])
        testing_params[param] = np.array([])
        val_params[param] = np.array([])

    # testing split is allways the last 0.1 percent
    # all the data between training split and 0.9 is not used
    # in order to use all the data use training split = 0.9

    my_items = list(data.items())
    # random.shuffle(my_items)
    for i, sample in enumerate(my_items):
        if i <= len(my_items) * training_split:
            reduced_sample = sample[1]['ps'][85:]
            if np.all(reduced_sample > 0.1):
                # if (np.max(reduced_sample) - np.min(reduced_sample)) > 100 and np.all(reduced_sample > 0.1):
                # if np.any(reduced_sample < 5) and np.all(reduced_sample > 0.1):  # and sample[1]['model params']['L_X'] > 38:
                for param in sample[1]['model params'].items():
                    training_params[param[0]] = np.append(training_params[param[0]], param[1])
                # training_params['tau'] =np.append(training_params['tau'], sample[1]['tau'])
                # training_params['x_HI'] = np.append(training_params['x_HI'], sample[1]['xH'])
                training_powerspectra += [reduced_sample]

        elif i <= len(data.items()) * (training_split + val_split):
            reduced_sample = sample[1]['ps'][85:]
            if np.all(reduced_sample > 0.1):
                # if (np.max(reduced_sample) - np.min(reduced_sample)) > 100 and np.all(reduced_sample > 0.1):
                # if np.any(reduced_sample < 5) and np.all(reduced_sample > 0.1):  # and sample[1]['model params']['L_X'] > 38:
                for param in sample[1]['model params'].items():
                    val_params[param[0]] = np.append(val_params[param[0]], param[1])
                # val_params['tau'] = np.append(val_params['tau'], sample[1]['tau'])
                # val_params['x_HI'] = np.append(val_params['x_HI'], sample[1]['xH'])
                val_powerspectra += [reduced_sample]

        elif i > len(data.items()) * (training_split + val_split):  # (training_split + val_split):
            reduced_sample = sample[1]['ps'][85::]
            if np.all(reduced_sample > 0.1):
                # if (np.max(reduced_sample) - np.min(reduced_sample)) > 100 and np.all(reduced_sample > 0.1):
                # if np.any(reduced_sample < 5) and np.all(reduced_sample > 0.1):  # and sample[1]['model params']['L_X'] > 38:
                for param in sample[1]['model params'].items():
                    testing_params[param[0]] = np.append(testing_params[param[0]], param[1])
                # testing_params['tau'] = np.append(testing_params['tau'], sample[1]['tau'])
                # testing_params['x_HI'] = np.append(testing_params['x_HI'], sample[1]['xH'])
                testing_powerspectra += [reduced_sample]

    # my_items = list(extra_data.items())
    # random.shuffle(my_items)
    # for i, sample in enumerate(my_items):
    #     if i <= len(extra_data.items()) * training_split:
    #         if np.any(sample[1]['ps'] > 1) and sample[1]['model params']['L_X'] > 38:
    #             for param in sample[1]['model params'].items():
    #                 training_params[param[0]] = np.append(training_params[param[0]], param[1])
    #             training_powerspectra += [sample[1]['ps'][30:]]
    #
    #     elif i <= len(extra_data.items()) * (training_split + val_split):
    #         if np.any(sample[1]['ps'] > 1) and sample[1]['model params']['L_X'] > 38:
    #             for param in sample[1]['model params'].items():
    #                 val_params[param[0]] = np.append(val_params[param[0]], param[1])
    #             val_powerspectra += [sample[1]['ps'][30:]]
    #
    #     elif i > len(extra_data.items()) * (training_split + val_split):
    #         if np.any(sample[1]['ps'] > 1) and sample[1]['model params']['L_X'] > 38:
    #             for param in sample[1]['model params'].items():
    #                 testing_params[param[0]] = np.append(testing_params[param[0]], param[1])
    #             testing_powerspectra += [sample[1]['ps'][30:]]

    testing_features = np.array(testing_powerspectra)
    features = np.array(training_powerspectra)
    val_features = np.array(val_powerspectra)
    k_range = data[0]['k'][30:89]

    print('model params: ', model_params)
    print('k modes: ', f'from {min(k_range)} to {max(k_range)} with {len(k_range)} k bins')
    print('training set size: ' + str(len(training_params['F_ESC10'])))
    print('features shape: ', features.shape)
    print('validation set size: ', val_features.shape[0])
    print('testing set size: ', testing_features.shape[0])
    return training_params, features, val_params, val_features, testing_params, testing_features, k_range, model_params


powerspectra, params, model_params, k_range = organize_data(data)

training_params, features, val_params, val_features, testing_params, testing_features = \
    divide_data(params, powerspectra, model_params, 0.85, 0.10)


# training_params, features, val_params, val_features, testing_params, testing_features, k_range, model_params = \
#     split_data(0.85, 0.1,
#                data, extra_data)
training_params['NU_X_THRESH'] = training_params['NU_X_THRESH'] / 1000
val_params['NU_X_THRESH'] = val_params['NU_X_THRESH'] / 1000
testing_params['NU_X_THRESH'] = testing_params['NU_X_THRESH'] / 1000

files = [training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range]
with open('/Users/hovavlazare/PycharmProjects/global_signal_model/log_ps_model_files/training_files.pk', 'wb') as f:
    pickle.dump(files, f)

with open('/Users/hovavlazare/PycharmProjects/global_signal_model/log_ps_model_files/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params, k_range = pickle.load(
        f)

# tmp = np.any(testing_features < 0, axis=1)
# a = np.sum(tmp)
# tmp1 = np.sum(features, axis=1)
# a1 = np.min(tmp1)
# arg = np.argmin(tmp1)
# f = features[arg,:]
# x=5


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
                      hidden_dims=[288, 512, 512, 288, 512, 1024], features_band=k_range, reg_factor=0.0, dropout_rate=0.0,
                      use_log=False, activation='linear', name='small_data_set_very_deep_emulator'
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

myEmulator.save('/Users/hovavlazare/PycharmProjects/global_signal_model/log_ps_model_files')
