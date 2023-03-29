import tensorflow as tf
import numpy as np
import h5py


class SignalClassifier:

    def __init__(self,
                 params_train_dict=None,
                 params_val_dict=None,
                 features_train=None,
                 features_val=None,
                 hidden_dims=None,
                 param_labels=None,
                 restore=False,
                 dropout_rate=0,
                 reg_factor=0,
                 activation='relu',
                 name='emulator',
                 files_dir=None
                 ):
        if restore:
            self.restore(files_dir, name)

        else:
            self.param_labels = param_labels
            self.hidden_dims = hidden_dims
            self.features_val = features_val
            self.features_train = features_train
            self.params_val_dict = params_val_dict
            self.params_train_dict = params_train_dict
            self.name = name

            self.params_train_arr = self.dict_to_ordered_arr_np(self.params_train_dict)

            self.tr_params_min = np.min(self.params_train_arr, axis=0)
            self.tr_params_max = np.max(self.params_train_arr, axis=0)

            self.params_train_arr = self.preprocess_params_arr(self.params_train_arr)
            self.params_val_arr = self.preprocess_params_dict(self.params_val_dict)

        self.NN = self.create_model(self.params_train_arr.shape[1], hidden_dims, self.features_train.shape[1],
                                    activation, dropout_rate, reg_factor, name)

    def create_model(self, input_dim, hidden_dims, out_dim, activation, dropout_rate, reg_factor, name):
        layers = []

        input_layer = tf.keras.Input(shape=(input_dim,))
        layers.append(input_layer)

        for dim in hidden_dims:

            layer1 = tf.keras.layers.Dense(dim, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=tf.keras.regularizers.l2(reg_factor),
                                           activation=activation)
            layers.append(layer1)
            if dropout_rate != 0.0:
                d_layer = tf.keras.layers.Dropout(rate=dropout_rate)
                layers.append(d_layer)

            layer3 = tf.keras.layers.BatchNormalization(trainable=True)
            layers.append(layer3)

        output_layer = tf.keras.layers.Dense(out_dim, activation='sigmoid')
        layers.append(output_layer)
        model = tf.keras.Sequential(layers, name=name)
        return model

    def dict_to_ordered_arr_np(self,
                               input_dict,
                               ):
        r"""
        Sort input parameters
        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted
        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.param_labels is not None:
            return np.stack([input_dict[k] for k in self.param_labels], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)

    def preprocess_params_arr(self, params_arr):
        params_arr -= self.tr_params_min
        params_arr /= (self.tr_params_max - self.tr_params_min)
        params_arr = params_arr * 2 - 1
        return params_arr

    def preprocess_params_dict(self, params_dict):
        params_arr = self.dict_to_ordered_arr_np(params_dict)
        return self.preprocess_params_arr(params_arr)

    def save(self, dir_path):
        self.NN.save(F'{dir_path}/{self.NN.name}.h5')
        h5f = h5py.File(f'{dir_path}/model_data.h5', 'w')
        h5f.create_dataset('tr_params_min', data=self.tr_params_min)
        h5f.create_dataset('tr_params_max', data=self.tr_params_max)
        h5f.create_dataset('param_labels', data=self.param_labels)
        h5f.close()

    def restore(self, dir_path, model_name):
        self.NN = tf.keras.models.load_model(f'{dir_path}/{model_name}.h5')
        h5f = h5py.File(f'{dir_path}/model_data.h5', 'r')
        self.tr_params_min = h5f['tr_params_min'][:]
        self.tr_params_max = h5f['tr_params_max'][:]
        self.param_labels = h5py.Dataset.asstr(h5f['param_labels'][:])[:]

    def train(self,
              initial_lr=0.01,
              batch_size=256,
              epochs=350,
              reduce_lr_factor=0.95,
              stop_patience_value=15,
              decay_patience_value=5,
              verbose=False
              ):

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=stop_patience_value, min_delta=1e-10, restore_best_weights=True, verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=decay_patience_value, factor=reduce_lr_factor,
            verbose=1, min_delta=5e-9, min_lr=1e-6)

        callbacks = [early_stopping_cb, reduce_lr_cb]

        self.NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                        metrics=[tf.keras.metrics.BinaryAccuracy()])

        history = self.NN.fit(x=self.params_train_arr,
                              y=self.features_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.params_val_arr, self.features_val),
                              validation_batch_size=batch_size,
                              callbacks=callbacks,
                              verbose=verbose,
                              )

        return history.history['loss'], history.history['val_loss']

    def predict(self, params_dict):
        params = self.preprocess_params_dict(params_dict)
        pred_features = self.NN.predict(params, verbose=False)
        return pred_features

    def evaluate(self, test_params_dict, logic_test):
        x_test = self.preprocess_params_dict(test_params_dict)
        test_scores = self.NN.evaluate(x_test, logic_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])
        return test_scores
