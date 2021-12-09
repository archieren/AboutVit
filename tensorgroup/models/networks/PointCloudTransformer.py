import numpy as np
import tensorflow as tf
from tensorflow.keras import (Sequential)
from tensorflow.keras.layers import (BatchNormalization,
                                    Concatenate,
                                    Dense, 
                                    Input,
                                    Layer,
                                    LeakyReLU,
                                    LayerNormalization,
                                    GlobalAveragePooling1D,
                                    ReLU)
from tensorflow.python.keras.engine.sequential import Sequential
from .layers.msa import (MultiHeadSelfAttention as MSA)

class LBR(Layer):
    def __init__(self, channels, repeats=2):
        super(LBR, self).__init__()
        # Dense作为Linear op
        self.net = Sequential([
                            Sequential([ Dense(channels, use_bias=False), BatchNormalization(), ReLU()])
                            for i in range(repeats)
                            ])


    def call(self, x):
        return self.net(x)

    def get_config(self):
        return super().get_config()

class StackTransformerLayer(Layer):
    def __init__(self, channels=256, num_heads=1):
        super(StackTransformerLayer, self).__init__()
        self.lbr = LBR(channels)
        self.msa1 = MSA(channels, num_heads)
        self.msa2 = MSA(channels, num_heads)
        self.msa3 = MSA(channels, num_heads)
        self.msa4 = MSA(channels, num_heads)
        self.concat = Concatenate(axis=-1)

    def call(self, x):
        x = self.lbr(x)
        x1 = self.msa1(x)
        x2 = self.msa2(x1)
        x3 = self.msa3(x2)
        x4 = self.msa4(x3)
        return self.concat([x1, x2, x3, x4])
    
    def get_config(self):
        return super().get_config()

def point_cloud_transformer():
    inputs = Input(shape=(1000, 3))
    outputs = StackTransformerLayer(channels=128)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
