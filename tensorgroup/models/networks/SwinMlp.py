import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, 
                                    Input,
                                    LayerNormalization,
                                    GlobalAveragePooling1D)
from .layers.patches import (Patches, PatchEncoder, PostPatchMerging)
from .layers.swin_mlp import BasicLayer

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}


def SwinMlp_classifier(
    image_size,
    patch_size=4,
    win_size=7,
    depths=[2, 2, 6, 2],
    num_classes = 1000,
    d_model = 96,
    num_heads=[3, 6, 12, 24],
    channels=3):
    inputs = Input(shape=( image_size, image_size, channels))
    patches = Patches(patch_size, channels)(inputs)
    num_patches = (image_size // patch_size) ** 2
    x = PatchEncoder(num_patches, d_model)(patches)
    for i in range(len(depths)):
        patches_shape = (image_size//patch_size//2**i, image_size//patch_size//2**i)
        if i < len(depths) - 1:
            x = PostPatchMerging(patches_shape , 
                            d_model*2**i, 
                            BasicLayer(d_model*2**i, patches_shape , depths[i], num_heads[i], win_size)
                            )(x)
        else:
            x = BasicLayer(d_model*2**i, patches_shape , depths[i], num_heads[i], win_size)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes, activation='softmax', name='logits')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

