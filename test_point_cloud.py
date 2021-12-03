import tensorgroup.models.utils.pointnet_utils as PU
import tensorgroup.models.networks.PointCloudTransformer as PCT
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Input,
    LayerNormalization,
)
import time

for _ in range(10):
    start = time.time()
    points = tf.random.uniform([10, 10000, 2], minval=1, maxval=1000)
    samples = PU.farthest_point_sample(points, 1000)
    elapsed = time.time() - start
    print(elapsed)


model = PCT.point_cloud_transformer()
model.summary()