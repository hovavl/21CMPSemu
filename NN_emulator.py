from typing import Optional
import h5py
import numpy as np
import tensorflow as tf



class emulator:
    def __init__(self,
                 params_train_dict=None,
                 params_val_dict=None,
                 features_train=None,
                 features_val=None,
                 param_labels=None,
                 hidden_dims=None,
                 features_band=None,
                 restore=False,
                 dropout_rate=0,
                 reg_factor=0,
                 use_log=False,
                 activation='relu',
                 name='emulator',
                 use_custom_layer = True,
                 use_batchNorm = True,
                 files_dir=None
                 ):
        if restore:
            self.restore(files_dir, name)

        else:
            self.features_band = features_band
            self.param_labels = param_labels
            self.features_val = features_val
            self.features_train = features_train
            self.params_train_dict = params_train_dict
            self.params_val_dict = params_val_dict

            self.params_train_arr = self.dict_to_ordered_arr_np(self.params_train_dict)

            self.tr_params_min = np.min(self.params_train_arr, axis=0)
            self.tr_params_max = np.max(self.params_train_arr, axis=0)

            self.params_train_arr = self.preprocess_params_arr(self.params_train_arr)
            self.params_val_arr = self.preprocess_params_dict(self.params_val_dict)
            if use_log:
                self.tr_features_mean = np.mean(np.log10(self.features_train), axis=0, dtype=np.float32)
                self.tr_features_std = np.std(np.log10(self.features_train), axis=0, dtype=np.float32)
            else:
                self.tr_features_mean = np.mean(self.features_train, axis=0, dtype=np.float32)
                self.tr_features_std = np.std(self.features_train, axis=0, dtype=np.float32)
            self.NN = self.create_model(self.params_train_arr.shape[1], hidden_dims, self.features_train.shape[1],
                                        activation, dropout_rate, reg_factor, name, use_custom_layer, use_batchNorm)

        self.use_log = use_log

    def preprocess_features(self, features):
        """
        preprocess features for training and validation by subtracting the mean of the training features and dividing
        by their std, and optionally take their log10 (important for power spectra)
         :param features: features to preprocess
         :return: preprocessed features
        """
        new_features = features.copy()
        if self.use_log:
            new_features = np.log10(new_features)
        # new_features -= self.tr_features_mean
        # new_features /= self.tr_features_std

        return new_features

    def postprocess_features(self, features):
        new_features = features.copy()
        # new_features = features * self.tr_features_std
        # new_features += self.tr_features_mean
        if self.use_log:
            new_features = 10 ** new_features
        return new_features

    def preprocess_params_arr(self, params_arr):
        params_arr -= self.tr_params_min
        params_arr /= (self.tr_params_max - self.tr_params_min)
        params_arr = params_arr * 2 - 1
        return params_arr

    def preprocess_params_dict(self, params_dict):
        params_arr = self.dict_to_ordered_arr_np(params_dict)
        return self.preprocess_params_arr(params_arr)

    def create_model(self, input_dim, hidden_dims, out_dim, activation, dropout_rate, reg_factor, name, use_custom_layer, use_BatchNorm):
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
            if use_custom_layer:
                layer2 = CustomLayer(units=dim, trainable=True)
                layers.append(layer2)
            if use_BatchNorm:
                layer3 = tf.keras.layers.BatchNormalization(trainable=True)
                layers.append(layer3)

        output_layer = tf.keras.layers.Dense(out_dim)
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

    def mre_loss(self):
        """
        max relative error: calculate the squared L2 norm divided bt the maximum feature amplitude
        :return: loss function
        """

        def loss_func(y_true, y_pred):
            y_real = y_true + tf.convert_to_tensor(self.tr_features_mean / self.tr_features_std)

            max_amp = tf.math.square(tf.math.reduce_max(tf.abs(y_real), axis=1, keepdims=False))

            loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred) / max_amp
            return loss

        return loss_func

    def L2_loss(self):
        """
        L2 loss function - mean squared error
        :return: loss function
        """

        def loss_func(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        return loss_func

    def APE_loss(self):
        """
                L2 loss function - mean absolute percentage error
                :return: loss function
                """

        def loss_func(y_true, y_pred):
            return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)

        return loss_func

    def train(self,
              loss_func_name='MRE',
              initial_lr=0.01,
              batch_size=256,
              epochs=350,
              reduce_lr_factor=0.95,
              stop_patience_value=15,
              decay_patience_value=5,
              verbose=False

              ):

        y_train = self.preprocess_features(self.features_train)
        y_val = self.preprocess_features(self.features_val)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=stop_patience_value, min_delta=1e-10, restore_best_weights=True, verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=decay_patience_value, factor=reduce_lr_factor,
            verbose=1, min_delta=5e-9, min_lr=1e-6)

        callbacks = [early_stopping_cb, reduce_lr_cb]

        if loss_func_name == 'MRE':
            loss_func = self.mre_loss()
        elif loss_func_name == 'L2':
            loss_func = self.L2_loss()
        elif loss_func_name == 'APE':
            loss_func = self.APE_loss()

        self.NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), loss=loss_func)

        history = self.NN.fit(x=self.params_train_arr,
                              y=y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.params_val_arr, y_val),
                              validation_batch_size=batch_size,
                              callbacks=callbacks,
                              verbose=verbose,
                              )

        return history.history['loss'], history.history['val_loss']

    def retrain(self,
                x_tr,
                y_tr,
                x_val,
                y_val,
                loss_func_name='MRE',
                initial_lr=0.001,
                batch_size=256,
                epochs=350,
                reduce_lr_factor=0.95,
                stop_patience_value=15,
                decay_patience_value=5,
                verbose=False
                ):

        x_tr_arr = self.preprocess_params_dict(x_tr)
        x_val_arr = self.preprocess_params_dict(x_val)

        y_train = self.preprocess_features(y_tr)
        y_val = self.preprocess_features(y_val)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=stop_patience_value, min_delta=1e-10, restore_best_weights=True, verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=decay_patience_value, factor=reduce_lr_factor,
            verbose=1, min_delta=5e-9, min_lr=1e-7)

        callbacks = [early_stopping_cb, reduce_lr_cb]

        if loss_func_name == 'MRE':
            loss_func = self.mre_loss()
        elif loss_func_name == 'L2':
            loss_func = self.L2_loss()
        elif loss_func_name == 'APE':
            loss_func = self.APE_loss()

        self.NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr), loss=loss_func)

        history = self.NN.fit(x=x_tr_arr,
                              y=y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_val_arr, y_val),
                              validation_batch_size=batch_size,
                              callbacks=callbacks,
                              verbose=verbose,
                              )

        return history.history['loss'], history.history['val_loss']




    def save(self, dir_path):
        self.NN.save(F'{dir_path}/{self.NN.name}.h5')

        h5f = h5py.File(f'{dir_path}/model_data.h5', 'w')
        h5f.create_dataset('tr_params_min', data=self.tr_params_min)
        h5f.create_dataset('tr_params_max', data=self.tr_params_max)
        h5f.create_dataset('tr_features_mean', data=self.tr_features_mean)
        h5f.create_dataset('tr_features_std', data=self.tr_features_std)
        h5f.create_dataset('param_labels', data=self.param_labels)
        h5f.create_dataset('features_band', data=self.features_band)
        h5f.close()

    def restore(self, dir_path, model_name):
        custom_object = {"loss_func": self.mre_loss(), 'CustomLayer': CustomLayer}
        self.NN = tf.keras.models.load_model(f'{dir_path}/{model_name}.h5', custom_objects=custom_object)
        h5f = h5py.File(f'{dir_path}/model_data.h5', 'r')
        self.tr_params_min = h5f['tr_params_min'][:]
        self.tr_params_max = h5f['tr_params_max'][:]
        self.tr_features_mean = h5f['tr_features_mean']
        self.tr_features_std = h5f['tr_features_std']
        self.param_labels = h5py.Dataset.asstr(h5f['param_labels'][:])[:]
        self.features_band = h5f['features_band'][:]

    def predict(self, params_dict):
        params = self.preprocess_params_dict(params_dict)
        pred_features = self.NN.predict(params, verbose=False)
        return self.postprocess_features(pred_features)

    def test_l2(self, test_params, test_features):
        pred = self.predict(test_params)
        losses = np.sqrt(np.mean(np.square(test_features - pred), axis=1))
        return losses, pred

    def test_MRE(self, test_params, test_features):
        losses, pred = self.test_l2(test_params, test_features)
        amp = np.max(np.abs(test_features), axis=1)
        return (losses / amp) * 100, pred

    def test_APE(self, test_params, test_features):
        pred = self.predict(test_params)
        loss = 100 * np.mean(np.abs((test_features - pred) / test_features), axis=1)

        return loss, pred


class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, alpha=None,
                 units=512,
                 trainable=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.trainable = trainable
        self.units = units
        self.alpha = alpha

    def build(self, input_shape):
        alpha_init = tf.keras.initializers.GlorotNormal()
        self.alpha = tf.Variable(
            dtype=tf.float32,
            initial_value=alpha_init(shape=(self.units,)),
            trainable=self.trainable,
            name="alpha")
        super().build(input_shape)

    def call(self, inputs):
        elem1 = tf.subtract(1.0, self.alpha)
        elem2 = tf.keras.activations.sigmoid(inputs)
        ptrs = tf.add(self.alpha, tf.math.multiply(elem1, elem2))
        return tf.math.multiply(inputs, ptrs)

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({
            "alpha": self.get_weights()[0] if self.trainable else self.alpha,
            "trainable": self.trainable,
            'units': self.units
        })
        return config

