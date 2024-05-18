import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('font', size=16)
from sklearn.preprocessing import MinMaxScaler
import warnings
import keras_tuner as kt
from keras import backend as K
from tensorflow import keras
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Method for creating sequences
def build_sequences(df, target_labels, window, stride, telescope):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    # convert dataset in ndarray
    temp_df = df.copy().values
    temp_label = df[target_labels].copy().values
    padding_len = len(df)%window

    if(padding_len != 0):
        # Compute padding length
        padding_len = window - len(df)%window
        padding = np.zeros((padding_len,temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding,df))
        padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding,temp_label))
        assert len(temp_df) % window == 0

    for idx in np.arange(0,len(temp_df)-window-telescope,stride):
        dataset.append(temp_df[idx:idx+window])
        labels.append(temp_label[idx+window:idx+window+telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

dataset = pd.read_csv('../input/dataset2/Training.csv')
target_labels = dataset.columns

# Split and Normalize data
X_train_raw = dataset[0:int(dataset.shape[0]*0.8)]
X_val_raw = dataset[int(dataset.shape[0]*0.8):int(dataset.shape[0]*0.9)]
X_test_raw = dataset[int(dataset.shape[0]*0.9):]
#X_min = X_train_raw.min()
#X_max = X_train_raw.max()
#X_train_raw = (X_train_raw-X_min)/(X_max-X_min)
#X_val_raw = (X_val_raw-X_min)/(X_max-X_min)
#X_test_norm_raw = (X_test_raw-X_min)/(X_max-X_min)
train_mean = X_train_raw.mean()
train_std = X_train_raw.std()
X_train_raw = (X_train_raw-train_mean)/train_std
X_val_raw = (X_val_raw-train_mean)/train_std
X_test_norm_raw = (X_test_raw-train_mean)/train_std


# Window, stride, telescope, batch and split
window = [3888]
#window = [864, 1296, 1728, 2160, 2592, 3456, 3888, 4320]
#window = [144, 288, 432, 576, 720, 864, 1008]
stride = [36, 72, 144]
#stride = [6, 18, 36, 72, 144]
telescope = 864
#batch = [64, 128, 256]
batch = [64]
# LR
learning_rate=1e-3

# Model
def build_CONV_LSTM_model(input_shape, output_shape):
    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    convlstm = tfkl.BatchNormalization()(input_layer)
    convlstm = tfkl.Conv1D(128, 6, padding='same', activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(convlstm)
    convlstm = tfkl.Conv1D(128, 12, padding='same', activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(convlstm)
    convlstm = tfkl.Conv1D(128, 24, padding='same', activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(convlstm)
    convlstm = tfkl.BatchNormalization()(convlstm)
    convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True, dropout=0.2, activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5)))(convlstm)
    convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True, dropout=0.2, activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5)))(convlstm)     
    convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True, dropout=0.2, activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5)))(convlstm)
    convlstm = tfkl.GlobalAveragePooling1D()(convlstm)
    convlstm = tfkl.Dropout(.2)(convlstm)
    convlstm = tfkl.BatchNormalization()(convlstm)
    dense = tfkl.Dense(256, activation='tanh', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(convlstm)
    dense = tfkl.Dropout(.2)(dense)
    dense = tfkl.BatchNormalization()(dense)
    dense = tfkl.Dense(output_shape[-1]*output_shape[-2], activation='linear', kernel_initializer=tfk.initializers.GlorotUniform(seed=seed), 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5))(dense)
    output_layer = tfkl.Reshape((output_shape[-2],output_shape[-1]))(dense)
    output_layer = tfkl.Conv1D(output_shape[-1], 1, padding='same')(output_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mae', rmse, lr_metric])

    # Return the model
    return model

# Cycle for window, stride, batch and split
for w in window:
    for s in stride:
        for b in batch:
                
            # Sequences
            X_train, y_train = build_sequences(X_train_raw, target_labels, w, s, telescope)
            X_val, y_val = build_sequences(X_val_raw, target_labels, w, s, telescope)
            X_test, y_test = build_sequences(X_test_raw, target_labels, w, s, telescope)
            X_test_norm, y_test_norm = build_sequences(X_test_norm_raw, target_labels, w, s, telescope)
               
            input_shape = X_train.shape[1:]
            output_shape = y_train.shape[1:]
               
            model = build_CONV_LSTM_model(input_shape, output_shape)
            model.summary()

            # Train the model
            history = model.fit(
                x = X_train,
                y = y_train,
                batch_size = b,
                epochs = 200,
                shuffle = False,
                validation_data=(X_val, y_val),
                callbacks = [
                    tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True),
                    tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=5, min_lr=1e-7)
                ]
            ).history

            model.save('CNN-BiLSTM_window'+str(w)+'_stride'+str(s)+'_batch'+str(b)+'_LR'+str(learning_rate)+'.h5')

            # Plot of results
            best_epoch = np.argmin(history['val_loss'])
            plt.figure(figsize=(17,4))
            plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
            plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
            plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
            plt.title('Mean Squared Error window'+str(w)+'_stride'+str(s)+'_batch'+str(b)+'_LR'+str(learning_rate))
            plt.legend()
            plt.grid(alpha=.3)
            plt.show()

            plt.figure(figsize=(17,4))
            plt.plot(history['mae'], label='Training accuracy', alpha=.8, color='#ff7f0e')
            plt.plot(history['val_mae'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
            plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
            plt.title('Mean Absolute Error window'+str(w)+'_stride'+str(s)+'_batch'+str(b)+'_LR'+str(learning_rate))
            plt.legend()
            plt.grid(alpha=.3)
            plt.show()
                
            plt.figure(figsize=(17,4))
            plt.plot(history['rmse'], label='Training loss', alpha=.8, color='#ff7f0e')
            plt.plot(history['val_rmse'], label='Validation loss', alpha=.9, color='#5a9aa5')
            plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
            plt.title('Root Mean Squared Error window'+str(w)+'_stride'+str(s)+'_batch'+str(b)+'_LR'+str(learning_rate))
            plt.legend()
            plt.grid(alpha=.3)
            plt.show()
                
            # Predict the test set 
            predictions = model.predict(X_test_norm)
                
            mean_squared_error = 0
            mean_absolute_error = 0
            # Post-processing on predictions
            for idx, p in enumerate(predictions):
                prediction = pd.DataFrame(p, columns = target_labels)
                #prediction = (prediction * (X_max-X_min)) + X_min
                prediction = (prediction * train_std) + train_mean
                prediction = prediction.to_numpy()
                #last_input = np.reshape(X_test[idx][-1], (1, 7))

                mean_squared_error = mean_squared_error + tfk.metrics.mse(y_test[idx].flatten(),prediction.flatten())
                mean_absolute_error = mean_absolute_error + tfk.metrics.mae(y_test[idx].flatten(),prediction.flatten())
                
            print('MSE')
            print(mean_squared_error/len(predictions))
            #print(tfk.metrics.mse(y_test.flatten(),predictions.flatten()))
            print('RMSE')
            print(np.sqrt(mean_squared_error/len(predictions)))
            print('MAE')
            print(mean_absolute_error/len(predictions))
