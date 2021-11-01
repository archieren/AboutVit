import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import LayerVersionSelector
import tensorflow_addons as tfa

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Embedding,
    Layer,
    LayerNormalization,
    Reshape
)

"""
## Implement the image patching layer! 
"""

class Patches(Layer):
    def __init__(self, patch_size, channels):
        super(Patches, self).__init__()
        # 用卷积作分片，也是可以的！
        # 对于图像类数据，可能更好，在底层，还是要利用CNN的特有能力。
        # 可以参看“Convolutional stem is all you need”，“CoAtNet: Marrying Convolution and Attention for All Data Sizes”等文！ 
        # 我们这儿就简单一些！
        patch_dims = patch_size**2*channels
        self.conv2d_as_patching = Conv2D(patch_dims, patch_size, patch_size)
        self.reshape = Reshape((-1, patch_dims))
    def call(self, images):
        """
        调用者应当保证image的shape合乎要求！
        """ 
        # (B, H, W, C)       
        patches = self.conv2d_as_patching(images)
        # (B, H//P, W//P, P*P*C)
        patches = self.reshape(patches)
        # (B, H*W // P**2, P*P*C)
        return patches

    def get_config(self):
        return super(Patches, self).get_config() 



"""
## Implement the patch encoding layer
The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        base_config = super(PatchEncoder, self).get_config()
        base_config['num_patches'] = self.num_patches
        return base_config

class ClassToken(Layer):
    def __init__(self, projection_dim):
        super(ClassToken, self).__init__()
        self.projection_dim = projection_dim
        self.cls_tokenizer = Embedding(
            input_dim=1, output_dim=projection_dim
        )

    def call(self, patch):
        batch_num = tf.shape(patch)[0]
        cls = tf.range(start=0, limit=1, delta=1)
        cls = self.cls_tokenizer(cls)
        cls = tf.broadcast_to(cls,(batch_num,1,self.projection_dim))
        return tf.concat((cls, patch), axis=1)

    def get_config(self):
        base_config = super(ClassToken, self).get_config()
        return base_config


class PatchMerging(Layer):
    '''
    Merge embedded patches.    
    '''
    def __init__(self, patches_shape, projection_dim):
        super().__init__()
        
        self.patches_shape = patches_shape
        self.projection_dim = projection_dim
        self.norm = LayerNormalization(epsilon=1e-5)
        self.reduction = Dense(2*projection_dim, use_bias=False)

    def call(self, x):
        
        H, W = self.patches_shape
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)
        
        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))
        
        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        
        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))
        x = self.norm(x)
        x = self.reduction(x)

        return x

class PostPatchMerging(Layer):

    def __init__(self, patches_shape, projection_dim, fn):
        super(PostPatchMerging, self).__init__()
        self.post = PatchMerging(patches_shape , projection_dim)
        self.fn = fn

    def call(self, x):
        return self.post(self.fn(x))
    
    def get_config(self):
        return super(PostPatchMerging, self).get_config()