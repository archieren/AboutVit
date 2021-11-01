import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.backend import transpose
from tensorflow.python.platform.tf_logging import _THREAD_ID_MASK


class Residual(tf.keras.layers.Layer):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x

    def get_config(self):
        return super().get_config()

class PreNorm(tf.keras.layers.Layer):

    def __init__(self, fn):
        super().__init__()
        self.norm = LayerNormalization(epsilon=1e-5)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))
    
    def get_config(self):
        return super().get_config()

class ChannelMlpBlock(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim):
        super(ChannelMlpBlock, self).__init__()
        self.net = tf.keras.Sequential([Dense(hidden_dim, activation='gelu'),
                                        Dense(dim)])

    def call(self, x):
        return self.net(x)
    
    def get_config(self):
        return super().get_config()

class TokenMlpBlock(tf.keras.layers.Layer):
    def __init__(self, patches_num, tokens_mlp_dim):
        super(TokenMlpBlock, self).__init__()
        self.net = tf.keras.Sequential([Dense(tokens_mlp_dim, activation='gelu'),
                                        Dense(patches_num)])

    def call(self, x):
        x = tf.transpose(x, perm=[0,2,1])
        x = self.net(x)
        x = tf.transpose(x, perm=[0,2,1])
        return x
    
    def get_config(self):
        return super().get_config()


class MlpMixerBlk(tf.keras.layers.Layer):
    def __init__(self, patches_num, dim, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixerBlk, self).__init__()
        self.token_fn = Residual(PreNorm(TokenMlpBlock(patches_num, tokens_mlp_dim)))
        self.channel_fn = Residual(PreNorm(ChannelMlpBlock(dim, channels_mlp_dim)))

    def call(self, x):
        return self.channel_fn(self.token_fn(x))
        
    def get_config(self):
        return super(MlpMixerBlk, self).get_config()