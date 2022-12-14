import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class GRUModel(tf.keras.Model):
    def __init__(self, input_shape=8, output_shape=1, hidden_shape=None):
        super(GRUModel, self).__init__()
        self.lstm_1 = layers.GRU(input_shape, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_1 = layers.Dropout(0.2)
        self.lstm_2 = layers.GRU(16, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_2 = layers.Dropout(0.2)
        self.dense = layers.Dense(16, activation='relu', kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_3 = layers.Dropout(0.2)
        self.ou = layers.Dense(1)

    def call(self, input_tensor, training=False):
        output =self.lstm_1(input_tensor)
        output = self.dropout_1(output)
        output =self.lstm_2(output)
        output = self.dropout_2(output)
        output = self.dense(output)
        output = self.dropout_3(output)
        output = self.ou(output)
        return output


class LSTMModel(tf.keras.Model):
    def __init__(self, input_shape=8, output_shape=1, hidden_shape=None):
        super(LSTMModel, self).__init__()
        self.lstm_1 = layers.LSTM(input_shape, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_1 = layers.Dropout(0.2)
        self.lstm_2 = layers.LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_2 = layers.Dropout(0.2)
        self.dense = layers.Dense(16, activation='relu', kernel_initializer='random_normal', kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout_3 = layers.Dropout(0.2)
        self.ou = layers.Dense(1)

    def call(self, input_tensor, training=False):
        output =self.lstm_1(input_tensor)
        output = self.dropout_1(output)
        output =self.lstm_2(output)
        output = self.dropout_2(output)
        output = self.dense(output)
        output = self.dropout_3(output)
        output = self.ou(output)
        return output