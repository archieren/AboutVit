import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Conv1D,
                                    Layer,
                                    LayerNormalization)
from tensorflow.keras.activations import softmax
from tensorflow.python.ops.control_flow_ops import group

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

class WindowMlp(Layer):
    def __init__(self, dim, win_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.spatial_mlp = Conv1D(num_heads*win_size**2, kernel_size=1, groups=num_heads)
    
    def call(self, x):
        _, N, C = x.get_shape().as_list()
        # N 应当是win_size**2
        head_dim = C//self.num_heads
        assert head_dim == self.head_dim , 'Dimension Error. Head_dim == Dim // Num_heads!'


        x = tf.reshape(x, shape=(-1, N, self.num_heads, head_dim))
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape =(-1, self.num_heads*N, head_dim))
        x = tf.transpose(x, perm=(0,2,1))
        x = self.spatial_mlp(x)
        x = tf.transpose(x, perm=(0,2,1))
        x = tf.reshape(x, shape=(-1, self.num_heads, N, head_dim))
        x = tf.transpose(x, perm=(0,2,1,3))
        x = tf.reshape(x, shape=(-1, N, self.num_heads*head_dim))
        return x

    def get_config(self):
        return super().get_config()

class Swin(Layer):
    def __init__(self, 
                dim, 
                patches_shape, 
                win_size, 
                shift_size,
                num_heads):
        super(Swin, self).__init__()
        # 输入将会是(B, L == H*W, C == dim)
        self.dim = dim 
        self.patches_shape = patches_shape # (H, W)
        self.ws = win_size # size of window
        self.ss = shift_size # size of window shift
        self.padding = [self.ws - self.ss, self.ss, self.ws - self.ss, self.ss]  # P_l,P_r,P_t,P_b

        self.spatial_mlp = WindowMlp(dim, win_size, num_heads)
        # Assertions
        assert 0 <= self.ss, 'shift_size >= 0 is required'
        assert self.ss < self.ws, 'shift_size < window_size is required'
        
        # <---!!!
        # Handling too-small patch numbers
        if min(self.patches_shape) < self.ws:
            self.ss = 0
            self.ws = min(self.patches_shape)
            
    def call(self, x):
        H, W = self.patches_shape
        B, L, C = x.get_shape().as_list()
        
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'
               
        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.ss > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = tf.pad(x, [[0, 0], [P_t, P_b], [P_l, P_r], [0, 0]], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # Window partition 
        x_windows = window_partition(shifted_x, self.ws)
        x_windows = tf.reshape(x_windows, shape=(-1, self.ws * self.ws, C))

        # Spatial mlp
        x_windows = self.spatial_mlp(x_windows)

        # Merge windows
        x_windows = tf.reshape(x_windows, shape=(-1, self.ws, self.ws, C))
        shifted_x = window_reverse(x_windows, _H, _W)

        # Reverse cyclic shift
        if self.ss > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :]
        else:
            x = shifted_x
            
        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H*W, C))

        return x

    def get_config(self):
        return super(Swin, self).get_config()

class SwinMlpBlock(Layer):
    def __init__(self, 
                dim, 
                patches_shape, 
                num_heads, 
                win_size, 
                shift_size, 
                mlp_dim):
        super(SwinMlpBlock, self).__init__()
        # 输入将会是(B, L == H*W, C == dim)
             
        self.res_swin_msa = Residual(Swin(  dim, 
                                            patches_shape,
                                            win_size,
                                            shift_size, 
                                            num_heads))
        self.res_mlp = Residual(PreNorm(Mlp(dim, mlp_dim)))
            

    def call(self, x):
        return self.res_mlp(self.res_swin_msa(x))

    def get_config(self):
        return super(SwinMlpBlock, self).get_config()


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
        self.blocks = tf.keras.Sequential([SwinMlpBlock(
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