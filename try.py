import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorgroup.models.networks.VisionTranformer as Vit
import tensorgroup.models.networks.VitMlp as VitMlp
import tensorgroup.models.networks.SwinTransfomer as SwinT
import tensorgroup.models.networks.SwinMlp as SwinM


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.
AUTOTUNE = tf.data.experimental.AUTOTUNE


IMAGE_SIZE=32
PATCH_SIZE=4 
NUM_LAYERS=8
NUM_HEADS=8
MLP_DIM=128

lr=1e-3
WEIGHT_DECAY=1e-4
BATCH_SIZE=64
epochs=60
#Load the dataset
ds = tfds.load("cifar10", as_supervised=True)
ds_train=(
    ds["train"]
    .batch(BATCH_SIZE)
    .shuffle(1024)
    .prefetch(AUTOTUNE)
)
ds_test = (
    ds["test"]
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# model = Vit.VIT_classifier(
#     image_size=IMAGE_SIZE,
#     patch_size=PATCH_SIZE,
#     num_layers=NUM_LAYERS,
#     num_classes=10,
#     d_model=64,
#     num_heads=NUM_HEADS,
#     mlp_dim=MLP_DIM,
#     channels=3,
#     dropout=0.1,
# )

# model = VitMlp.VitMlp_classifier(
#     image_size=IMAGE_SIZE,
#     patch_size=PATCH_SIZE,
#     num_classes=10,
#     d_model = 128,
#     num_blk = 8,
#     tokens_mlp_dim = 256, 
#     channels_mlp_dim = 256,
#     channels=3
# )

model = SwinT.SwinTranfomer_classifier(32, 2, 2, depths=[2,2,6,2], num_classes=10, d_model=36, num_heads=[3, 6, 12, 24])
# model = SwinM.SwinMlp_classifier(32, 2, 2, depths=[2,2,6,2], num_classes=10, d_model=36, num_heads=[3, 6, 12, 24])

model.summary()

checkpoint_dir = os.path.join(os.getcwd(), 'work', 'ckpt')
saved_model_dir = os.path.join(os.getcwd(), 'work', 'sm')

model.compile(  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # We have normalized the network output!
                optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),],
)

checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
# early_stop = tf.keras.callbacks.EarlyStopping(patience=10),
mcp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
#                                                  factor=0.1, 
#                                                  patience=3, 
#                                                  verbose=0, 
#                                                  mode='auto',
# min_delta=0.0001, cooldown=0, min_lr=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    model.load_weights(latest)
model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=epochs,
    callbacks=[mcp],
)

model.save(os.path.join(saved_model_dir, "vit.h5"))


for image, target in ds_test.take(1):
    print(image.shape)
    y = model.predict(image)
    print(np.argmax(y, axis=1), target)