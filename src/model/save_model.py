import joblib

def load_model(path):
    return joblib.load(path)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

# Custom Attention Layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)         # (batch_size, seq_len, 1)
        a = K.softmax(e, axis=1)                           # (batch_size, seq_len, 1)
        output = inputs * a                                # (batch_size, seq_len, features)
        return K.sum(output, axis=1)                       # (batch_size, features)

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config

if __name__ == "__main__":
    try:
        model = load_model(r'models/gesture_model.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
