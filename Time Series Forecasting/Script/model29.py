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

def get_lr_metric(optimizer)
    def lr(y_true, y_pred)
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

def rmse(y_true, y_pred)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Method for creating sequences
def build_sequences(value, window, stride, telescope)
    X=[]
    y=[]

    for idx in np.arange(0,len(value)-window-telescope,stride)
        X.append(value[idxidx+window])
        y.append(value[idx+windowidx+window+telescope])

    X = np.array(X)
    y = np.array(y)

    return X, y

def show_heatmap(data, method)
    plt.matshow(data.corr(method=method))
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(Feature Correlation Heatmap, fontsize=14)
    plt.show()

dataset = pd.read_csv('..inputdataset2Training.csv')
target_labels = dataset.columns
#show_heatmap(dataset, 'spearman')

# Normalization
scaler = MinMaxScaler()
dataset = dataset.to_numpy()
train = dataset[0int(len(dataset)0.9)]
val = dataset[int(len(dataset)0.9)]
scaler.fit(train)
train = scaler.transform(train)
val = scaler.transform(val)

# Parameters
window = [864, 1152, 1440, 1728, 2016, 2304, 2592, 2880, 3168, 3456, 3744, 4032, 4320, 4608, 4896, 5184] 
telescope= [864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864, 864]
stride = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
batch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

# Model
def build_model(input_shape, output_shape)

    n_hidden = 128

    input_train = tfkl.Input(shape=(X_train.shape[1], X_train.shape[2]))

    encoder_last_h1, encoder_last_h2, encoder_last_c = tfkl.LSTM(n_hidden, activation='tanh', dropout=0.2, recurrent_dropout=0.2,
                                                                 return_sequences=False, return_state=True)(input_train)

    encoder_last_h1 = tfkl.BatchNormalization()(encoder_last_h1)
    encoder_last_c = tfkl.BatchNormalization()(encoder_last_c)

    decoder = tfkl.RepeatVector(y_train.shape[1])(encoder_last_h1)
    decoder = tfkl.LSTM(n_hidden, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_state=False, return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])

    out = tfkl.TimeDistributed(tfkl.Dense(y_train.shape[2]))(decoder)

    model = tfk.Model(inputs=input_train, outputs=out)
    model.summary()

    # Compile the model
    optimizer=tfk.optimizers.Adam(1e-3)
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mae', rmse, get_lr_metric(optimizer)])

    # Return the model
    return model

for i in range(len(window))

    # Sequences
    X_train, y_train = build_sequences(train, window[i], stride[i], telescope[i])
    X_val, y_val = build_sequences(val, window[i], stride[i], telescope[i])

    # Train the modelù
    model = build_model(X_train.shape[1], y_train.shape[1])
    history = model.fit(
        x = X_train,
        y = y_train,
        batch_size = batch[i],
        epochs = 10,
        shuffle = False,
        validation_data=(X_val, y_val),
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True),
                     tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=5, min_lr=1e-7)]
    ).history

    # Save model
    model.save('model'+str(i)+'.h5')

    # Plot of results
    best_epoch = np.argmin(history['val_loss'])
    plt.figure(figsize=(17,4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Squared Error window')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17,4))
    plt.plot(history['mae'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_mae'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Absolute Error window')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17,4))
    plt.plot(history['rmse'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_rmse'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Root Mean Squared Error window')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    # Test
    X_test = dataset[len(dataset)-telescope[i]-window[i]len(dataset)-telescope[i]]
    X_test = scaler.transform(X_test)
    y_test = dataset[len(dataset)-telescope[i]len(dataset)]
    prediction = model.predict(X_test.reshape((1, 3456, 7)))

    prediction = scaler.inverse_transform(prediction.reshape((864,7)))
    print(tfk.metrics.mse(y_test.flatten(),prediction.flatten()))
