import numpy as np
import tensorflow as tf

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    这是一个用tf来进行点云采样的算法，凑和用。
    """
    B, N, _ = xyz.get_shape().as_list()
    sample_ids = tf.zeros([B, npoint], dtype=tf.int32)
    
    # 各点到已采样点集的距离，初始值为尽量大！注意这个数据结构，理解问题的关键。
    distance_to_sampled_points = tf.ones([B, N]) * 1e10
    
    # 当前采集到的点， 初始指为随机指派！
    current_sample = tf.random.uniform([B], maxval=N, dtype=tf.int32)
    
    # 辅助数据， 用于构造u_indices,来更新sample_ids。
    batch_indices = tf.range(0, B, dtype=tf.int32)
    current_indices = tf.zeros([B], dtype=tf.int32)

    for i in range(npoint): # 这是低效的来源！
        # 添加第i个采集到的点       
        u_indices = tf.stack([batch_indices, current_indices], -1)
        sample_ids = tf.tensor_scatter_nd_update(sample_ids, u_indices, current_sample)
        # 采集下一个样本点
        centroid = tf.gather(xyz, current_sample, axis=1, batch_dims=1)
        centroid = tf.reshape(centroid, [B, 1, -1])
        dist = tf.math.reduce_sum((xyz - centroid) ** 2, -1)
        distance_to_sampled_points = tf.math.minimum(distance_to_sampled_points, dist)
        current_sample = tf.math.argmax(distance_to_sampled_points, axis=-1, output_type=tf.int32)
        #
        current_indices = current_indices + 1
    # return tf.gather(xyz, tf.stack(centroids,axis=-1),  axis=1, batch_dims=1)
    return tf.gather(xyz, sample_ids ,  axis=1, batch_dims=1)