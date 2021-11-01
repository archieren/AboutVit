import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    Softmax
)

from .layers.mlp_mixer import MlpMixerBlk
from .layers.patches import Patches, PatchEncoder

def getClassToken(x):
    return x[:, 0]

def VitMlp_classifier(
    image_size,
    patch_size,
    num_classes,
    d_model,
    num_blk,
    tokens_mlp_dim, 
    channels_mlp_dim,
    channels=3
):
    inputs = Input(shape=( image_size, image_size, channels))
    patches = Patches(patch_size, channels)(inputs)
    num_patches = (image_size // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, d_model)(patches)
    # encoded_patches = ClassToken(d_model)(encoded_patches)
    # num_patches += 1

    for _ in range(num_blk):
        encoded_patches = MlpMixerBlk(num_patches, d_model, tokens_mlp_dim, channels_mlp_dim)(encoded_patches)

    cls_token = GlobalAveragePooling1D()(encoded_patches)  # TODO verify this global average pool is correct choice here

    cls_token = LayerNormalization(name='pre_head_layer_norm')(cls_token)
    logits= Dense(num_classes, name='head', activation='softmax')(cls_token)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model