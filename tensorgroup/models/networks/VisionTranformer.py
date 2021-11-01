import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Input,
    LayerNormalization,
)

from .layers.msa import TransformerBlock, FeedForward
from .layers.patches import Patches, PatchEncoder, ClassToken

def getClassToken(x):
    return x[:, 0]

def VIT_classifier(
    image_size,
    patch_size,
    num_layers,
    num_classes,
    d_model,
    num_heads,
    mlp_dim,
    channels=3,
    dropout=0.1
):
    inputs = Input(shape=( image_size, image_size, channels))
    patches = Patches(patch_size, channels)(inputs)
    num_patches = (image_size // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, d_model)(patches)
    encoded_patches = ClassToken(d_model)(encoded_patches)
    for _ in range(num_layers):
        encoded_patches = TransformerBlock(d_model, num_heads, mlp_dim, dropout)(encoded_patches)

    # 这个地方是个按应用来灵活处理的！
    cls_token = LayerNormalization(epsilon=1e-6)(getClassToken(encoded_patches))
    # Add MLP.
    features = FeedForward(1024, 2048)(cls_token)
    # # Classify outputs.
    logits = Dense(num_classes, activation='softmax')(features)
    # # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model