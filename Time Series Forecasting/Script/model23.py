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


# Method for creating sequences
def build_sequences(df, target_labels, window, stride, telescope):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    # convert dataset in ndarray
    temp_df = df.copy().values
    temp_label = df[target_labels].copy().values
    padding_len = len(df) % window

    if (padding_len != 0):
        # Compute padding length
        padding_len = window - len(df) % window
        padding = np.zeros((padding_len, temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding, df))
        padding = np.zeros((padding_len, temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding, temp_label))
        assert len(temp_df) % window == 0

    for idx in np.arange(0, len(temp_df) - window - telescope, stride):
        dataset.append(temp_df[idx:idx + window])
        labels.append(temp_label[idx + window:idx + window + telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


dataset = pd.read_csv('../input/dataset2/Training.csv')
target_labels = dataset.columns

# Normalize data
X_train_raw = dataset[0:int(dataset.shape[0] * 0.8)]
X_val_raw = dataset[int(dataset.shape[0] * 0.8):]
X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train_raw = (X_train_raw - X_min) / (X_max - X_min)
X_val_raw = (X_val_raw - X_min) / (X_max - X_min)

# Window, stride, telescope, batch and split
# window = [108, 216, 432, 864, 2592, 4320, 6048]
window = [864]
# stride = [2, 6, 12, 18, 24, 36]
stride = [12]
telescope = 864
# batch = [64, 128, 256]
batch = [128]
# split =[0.1, 0.2]
split = [0.2]


# Model
def build_CONV_LSTM_model(input_shape, output_shape):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    convlstm = tfkl.BatchNormalization()(input_layer)
    convlstm = tfkl.LSTM(64, return_sequences=True, dropout=0.3)(convlstm)
    convlstm = tfkl.LSTM(64, return_sequences=True, dropout=0.3)(convlstm)
    convlstm = tfkl.BatchNormalization()(convlstm)
    convlstm = tfkl.Conv1D(64, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.Conv1D(128, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.Conv1D(256, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.BatchNormalization()(convlstm)
    convlstm = tfkl.LSTM(64, return_sequences=True, dropout=0.3)(convlstm)
    convlstm = tfkl.LSTM(64, return_sequences=True, dropout=0.3)(convlstm)
    convlstm = tfkl.BatchNormalization()(convlstm)
    convlstm = tfkl.GlobalAveragePooling1D()(convlstm)
    dense = tfkl.Dense(output_shape[-1] * output_shape[-2], activation='relu')(convlstm)
    output_layer = tfkl.Reshape((output_shape[-2], output_shape[-1]))(dense)
    output_layer = tfkl.Conv1D(output_shape[-1], 1, padding='same')(output_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(1e-2), metrics=['mae'])
    model.summary()

    # Return the model
    return model


# Cycle for window, stride, batch and split
for w in window:
    for s in stride:
        for b in batch:
            for sp in split:
                # Sequences
                X_train, y_train = build_sequences(X_train_raw, target_labels, w, s, telescope)
                X_val, y_val = build_sequences(X_val_raw, target_labels, w, s, telescope)

                input_shape = X_train.shape[1:]
                output_shape = y_train.shape[1:]
                print(input_shape)

                model = build_CONV_LSTM_model(input_shape, output_shape)
                # model.summary()

                # Train the model
                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=b,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[
                        tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25,
                                                    restore_best_weights=True),
                        tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5,
                                                        min_lr=1e-7)
                    ]
                ).history

                model.save('CNN-BiLSTM_window' + str(w) + '_stride' + str(s) + '_batch' + str(b) + '_split' + str(
                    sp) + '_Model4.h5')

                # Plot of results
                best_epoch = np.argmin(history['val_loss'])
                plt.figure(figsize=(17, 4))
                plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
                plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
                plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
                plt.title(
                    'Mean Squared Error window' + str(w) + '_stride' + str(s) + '_batch' + str(b) + '_split' + str(
                        sp) + '_Model4')
                plt.legend()
                plt.grid(alpha=.3)
                plt.show()

                plt.figure(figsize=(17, 4))
                plt.plot(history['mae'], label='Training accuracy', alpha=.8, color='#ff7f0e')
                plt.plot(history['val_mae'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
                plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
                plt.title(
                    'Mean Absolute Error window' + str(w) + '_stride' + str(s) + '_batch' + str(b) + '_split' + str(
                        sp) + '_Model4')
                plt.legend()
                plt.grid(alpha=.3)
                plt.show()
