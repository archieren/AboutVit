import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Layer,
                                    LayerNormalization)
from tensorflow.keras.activations import softmax

def window_partition(x, win_size):
    """
    Args:
        x: (B, H, W, C)
        win_size (int): window size
    Returns:
        windows: (B*(H//win_size)*(W//win_size), win_size, win_size, C)
    """
    _, H, W, C = x.get_shape().as_list()
    
    # Subset patches to windows of patches
    x = tf.reshape(x, shape=(-1, H//win_size, win_size, W//win_size, win_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    
    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, win_size, win_size, C))
    
    return windows

def window_reverse(x, H, W):
    """
    Args:
        x: (B*(H//win_size)*(W//win_size), win_size, win_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    _, win_h, win_w, C = x.get_shape().as_list()

    x = tf.reshape(x, shape=(-1, H//win_h, W//win_w, win_h, win_w, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    
    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))
    
    return x

def gen_rpi(win_h, win_w):
    coords_h = np.arange(win_h)
    coords_w = np.arange(win_w)
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
    coords = np.stack(coords_matrix)
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w- 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index

def gen_label(ws, ss, H, W):
    h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
    label_array = np.zeros((1, H, W, 1))
    
    ## initialization
    ## 怎么来表述这个算法呢？
    label = 0
    for h in h_slices:
        for w in w_slices:
            label_array[:, h, w, :] = label
            # 同一个区域里的patch，具有相同的label。
            # 同一个区域里的patch，如果分到某Window，则具有相同的label！！
            # 那么后面， 就可以以此来生成attention mask!
            label += 1
    return label_array

class WindowAttention(Layer):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True):
        super().__init__()
        # 输入将是(B, N)
        self.dim = dim # number of input dimensions
        self.win_size = win_size # size of the attention window
        self.num_heads = num_heads # number of self-attention heads
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # query scaling factor
        
        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.proj = Dense(dim)

    def build(self, input_shape):
        
        # zero initialization
        num_window_elements = (2*self.win_size - 1) * (2*self.win_size - 1)
        self.relative_position_bias_table = self.add_weight('attn_pos',
                                                            shape=(num_window_elements, self.num_heads),
                                                            initializer=tf.initializers.Zeros(), 
                                                            trainable=True)
        
        # Indices of relative positions
        relative_position_index = gen_rpi(self.win_size, self.win_size)
        
        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False, name='attn_pos_ind')
        
        self.built = True

    def call(self, x, mask=None):
        
        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C//self.num_heads
        
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        
        # Query rescaling
        q = q * self.scale
        
        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)
        
        # Shift window
        win_area = self.win_size * self.win_size
        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias, shape=(win_area, win_area, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)
               
        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        
        # Linear projection
        x_qkv = self.proj(x_qkv)
                
        return x_qkv

class Mlp(Layer):
    """
    The so called mlp part!
    """

    def __init__(self, dim, hidden_dim):
        super(Mlp, self).__init__()
        self.net = Sequential([Dense(hidden_dim, activation='gelu'),
                               Dense(dim)])

    def call(self, x):
        return self.net(x)
    
    def get_config(self):
        return super(Mlp, self).get_config()


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

class Swin(Layer):
    def __init__(self, 
                dim, 
                patches_shape, 
                win_size, 
                shift_size,
                w_or_sw_msa):
        super(Swin, self).__init__()
        # 输入将会是(B, L == H*W, C == dim)
        self.dim = dim 
        self.patches_shape = patches_shape # (H, W)
        self.ws = win_size # size of window
        self.ss = shift_size # size of window shift
        self.attn = w_or_sw_msa
        # Assertions
        assert 0 <= self.ss, 'shift_size >= 0 is required'
        assert self.ss < self.ws, 'shift_size < window_size is required'
        
        # <---!!!
        # Handling too-small patch numbers
        if min(self.patches_shape) < self.ws:
            self.ss = 0
            self.ws = min(self.patches_shape)
            
    def build(self, input_shape):
        if self.ss > 0:
            H, W = self.patches_shape
            # 我改了名，以表明我对算法的理解！
            label_array = tf.convert_to_tensor(gen_label(self.ws, self.ss, H, W)) # (1, H, W,1)           
            # mask array to windows
            label_into_windows = window_partition(label_array, self.ws)  
            # (H*W//ws**2, ws, ws, 1)
            label_into_windows = tf.reshape(label_into_windows, shape=[-1, self.ws * self.ws]) 
            # (H*W//ws**2, ws**2)
            attn_mask = tf.expand_dims(label_into_windows, axis=1) - tf.expand_dims(label_into_windows, axis=2) 
            # (H*W//ws**2, 1, ws**2)-(H*W//ws**2, ws**2,1) == (H*W//ws**2, ws**2, ws**2)
            # 此时的结果就是，窗口中，具有相同标签的的patches才具有相关性！
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.patches_shape
        B, L, C = x.get_shape().as_list()
        
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'
               
        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.ss > 0:
            shifted_x = tf.roll(x, shift=[-self.ss, -self.ss], axis=[1, 2])
        else:
            shifted_x = x

        # Window partition 
        x_windows = window_partition(shifted_x, self.ws)
        x_windows = tf.reshape(x_windows, shape=(-1, self.ws * self.ws, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.ws, self.ws, C))
        shifted_x = window_reverse(attn_windows, H, W)

        # Reverse cyclic shift
        if self.ss > 0:
            x = tf.roll(shifted_x, shift=[self.ss, self.ss], axis=[1, 2])
        else:
            x = shifted_x
            
        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H*W, C))

        return x

    def get_config(self):
        return super(Swin, self).get_config()

class SwinTransformerBlock(Layer):
    def __init__(self, 
                dim, 
                patches_shape, 
                num_heads, 
                win_size, 
                shift_size, 
                mlp_dim,
                qkv_bias=True):
        super(SwinTransformerBlock, self).__init__()
        # 输入将会是(B, L == H*W, C == dim)
             
        self.res_swin_msa = Residual(PreNorm(Swin(  dim, 
                                                    patches_shape,
                                                    win_size,
                                                    shift_size, 
                                                    WindowAttention(dim, win_size, num_heads, qkv_bias))))
        self.res_mlp = Residual(PreNorm(Mlp(dim, mlp_dim)))
            

    def call(self, x):
        return self.res_mlp(self.res_swin_msa(x))

    def get_config(self):
        return super(SwinTransformerBlock, self).get_config()


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, 
                dim, 
                patches_shape, 
                depth, 
                num_heads, 
                window_size):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(
                                            dim=dim, 
                                            patches_shape=patches_shape,
                                            num_heads=num_heads, 
                                            win_size=window_size,
                                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                                            mlp_dim = dim*4) 
                                            for i in range(depth)])

    def call(self, x):
        x = self.blocks(x)
        return x

    def get_config(self):
        return super(BasicLayer, self).get_config()