import tensorflow as tf

from tensorflow.keras import (Sequential)
from tensorflow.keras.layers import (
    Dense,
    Layer,
    LayerNormalization,
)

class MultiHeadSelfAttention(Layer):
    def __init__(self, dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {dim} should be divisible by number of heads = {num_heads}"
            )
        self.head_dim = dim // num_heads
        #attention takes three inputs: queries, keys, and values,
        self.query_dense = Dense(dim)
        self.key_dense = Dense(dim)
        self.value_dense = Dense(dim)
        self.combine_heads = Dense(dim)

    def attention(self, query, key, value):
            #use the product between the queries and the keys 
            #to know "how much" each element is the sequence is important with the rest
            score = tf.matmul(query, key, transpose_b=True)
            dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
            #resulting vector, score is divided by a scaling factor based on the size of the embedding
            #scaling fcator is square root of the embeding dimension
            scaled_score = score / tf.math.sqrt(dim_key)
            #the attention scaled_score is then softmaxed
            weights = tf.nn.softmax(scaled_score, axis=-1)
            #Attention(Q, K, V ) = softmax[(QK)/√dim_key]V
            output = tf.matmul(weights, value)
            return output, weights
    
    def separate_heads(self, x, batch_size):
            x = tf.reshape(
                x, (batch_size, -1, self.num_heads, self.head_dim)
            )
            return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
            
            batch_size = tf.shape(inputs)[0]
            #MSA takes the queries, keys, and values  as input from the   
            #previous layer and projects them using the 3 linear layers.
            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)
            query = self.separate_heads(query, batch_size)
            key = self.separate_heads(key, batch_size)
            value = self.separate_heads(value, batch_size)
            attention, weights = self.attention(query, key, value)
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(
                attention, (batch_size, -1, self.dim)
            )
            #self attention of different heads are concatenated  
            output = self.combine_heads(concat_attention)
            return output
    
    def get_config(self):
        return super(MultiHeadSelfAttention, self).get_config()

class Residual(Layer):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x

    def get_config(self):
        return super().get_config()

class PreNorm(Layer):

    def __init__(self, fn):
        super().__init__()
        self.norm = LayerNormalization(epsilon=1e-5)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))
    
    def get_config(self):
        return super().get_config()

class FeedForward(Layer):
    """
    The so called mlp part!
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = Sequential([Dense(hidden_dim, activation='gelu'),
                                        Dense(dim)])

    def call(self, x):
        return self.net(x)
    
    def get_config(self):
        return super().get_config()


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        """
        这个DroupOut有用吗？
        """
        super(TransformerBlock, self).__init__()
        self.res_msa = Residual(PreNorm(MultiHeadSelfAttention(embed_dim, num_heads)))
        self.res_mlp = Residual(PreNorm(FeedForward(embed_dim, hidden_dim)))
    def call(self, inputs):
        return self.res_mlp(self.res_msa(inputs))

    def get_config(self):
        return super(TransformerBlock, self).get_config()