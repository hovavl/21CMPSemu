import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from Classifier import SignalClassifier

data = pickle.load(
    open('/Users/hovavlazare/GITs/21CMPSemu training data/pd_training_data_mini/samples_29-04-23_z=10.4.pk', 'rb'))


def classify_signal(signal):
    minimum = np.min(signal)
    maximum = np.max(signal)
    if maximum - minimum > 1 and np.all(signal > 0.5):
        return 1
    else:
        return 0


def organize_data(data):
    params = []
    logics = []
    class_counter = {}
    for i in range(2):
        class_counter[i] = 0
    model_params = list(data[0]['model params'].keys())
    # model_params += ['tau', 'x_HI']
    my_items = list(data.items())
    # random.shuffle(my_items)
    counter = 0
    for i, sample in enumerate(my_items):
        counter += 1
        if i <= len(my_items):
            reduced_sample = sample[1]['ps'][30:]
            class_num = classify_signal(reduced_sample)
            if class_counter[class_num] < 10000:
                class_counter[class_num] += 1
                params += [list(sample[1]['model params'].values())]
                logics += [class_num]
            if np.all(np.array(list(class_counter.values())) > 10000):
                break
    logics = np.array(logics)
    params = np.array(params)
    return logics, params, model_params


def divide_data(params, logics, model_params, tr_split, val_split):
    all_data = list(zip(params, logics))
    random.shuffle(all_data)
    s_params, s_logics = zip(*all_data)
    s_params = np.array(s_params)
    s_logics = np.array(s_logics)
    s_logics = np.reshape(s_logics, (s_logics.shape[0], 1))
    tr_params = s_params[:int(s_params.shape[0] * tr_split), :]
    tr_logics = s_logics[:int(s_logics.shape[0] * tr_split), :]
    val_params = s_params[int(s_params.shape[0] * tr_split) + 1:int(s_params.shape[0] * (tr_split + val_split)), :]
    val_logics = s_logics[int(s_logics.shape[0] * tr_split) + 1:int(
        s_logics.shape[0] * (tr_split + val_split)), :]
    test_params = s_params[int(s_params.shape[0] * (tr_split + val_split)) + 1:, :]
    test_logics = s_logics[int(s_logics.shape[0] * (tr_split + val_split)) + 1:, :]


    tr_par_dict = {}
    val_par_dict = {}
    test_par_dict = {}
    for j, par_name in enumerate(model_params):
        tr_par_dict[par_name] = tr_params[:, j]
        val_par_dict[par_name] = val_params[:, j]
        test_par_dict[par_name] = test_params[:, j]
    return tr_par_dict, tr_logics, val_par_dict, val_logics, test_par_dict, test_logics


powerspectra, params, model_params = organize_data(data)

training_params, features, val_params, val_features, testing_params, testing_features = \
    divide_data(params, powerspectra, model_params, 0.85, 0.10)

training_params['NU_X_THRESH'] = training_params['NU_X_THRESH'] / 1000
val_params['NU_X_THRESH'] = val_params['NU_X_THRESH'] / 1000
testing_params['NU_X_THRESH'] = testing_params['NU_X_THRESH'] / 1000

files = [training_params, features, val_params, val_features, testing_params, testing_features, model_params]
with open('/mini_halos/mini_halos_NN/classifier_model_files_10-4/training_files.pk', 'wb') as f:
    pickle.dump(files, f)

with open('/mini_halos/mini_halos_NN/classifier_model_files_10-4/training_files.pk', 'rb') as f:
    training_params, features, val_params, val_features, testing_params, testing_features, model_params = pickle.load(
        f)

myClassifier = SignalClassifier(params_train_dict=training_params, params_val_dict=val_params,
                                features_train=features,
                                features_val=val_features,
                                param_labels=model_params,
                                hidden_dims=[128, 512, 256],
                                reg_factor=0.00,
                                dropout_rate=0.1,
                                activation='relu', name='classify_NN__mini_104',
                                )

print(myClassifier.NN.summary())
train_loss, val_loss = myClassifier.train(reduce_lr_factor=0.5, batch_size=512, verbose=True,
                                          epochs=500, decay_patience_value=10, stop_patience_value=30)

plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

test_scr = myClassifier.evaluate(testing_params, testing_features)
predictions = myClassifier.predict(testing_params)
myClassifier.save('/Users/hovavlazare/GITs/21CMPSemu/mini_halos_NN/classifier_model_files_10-4')

x = 1
