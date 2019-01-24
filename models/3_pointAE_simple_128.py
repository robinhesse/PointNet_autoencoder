import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

from munkres import munkres
from scipy.spatial import distance
from tensorflow.python.framework import ops
import itertools
from scipy.optimize import linear_sum_assignment


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    goal_pcs = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, goal_pcs, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    print(point_cloud.get_shape().as_list())
    input_image = tf.expand_dims(point_cloud, -1)
    print(input_image.get_shape().as_list())
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    print(net.get_shape().as_list())
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    print(net.get_shape().as_list())

    #with tf.variable_scope('transform_net2') as sc:
    #    transform = feature_transform_net(net, is_training, bn_decay, K=64)
    #end_points['transform'] = transform
    #net_transformed = tf.matmul(tf.squeeze(net), transform)
    #net_transformed = tf.expand_dims(tf.squeeze(net), [2])


    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay) # remember to use net_transformed here
    print(net.get_shape().as_list())    
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    print(net.get_shape().as_list())    
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    print(net.get_shape().as_list())
    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    print(net.get_shape().as_list())    


    net3 = tf.reshape(net, [batch_size, -1])
    encoding = net3
    #print(net3.get_shape().as_list())
    net3 = tf_util.fully_connected(net3, 2048, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    #print(net3.get_shape().as_list())
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    nnet3 = tf_util.fully_connected(net3, 3096, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    print(net3.get_shape().as_list())
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp2')
    #net3 = tf_util.fully_connected(net3, 768, activation_fn=None, scope='fc3')
    #print(net3.get_shape().as_list())
    #net3 = tf.reshape(net3, [batch_size, 256,3])
    #print(net3.get_shape().as_list())
    net3 = tf_util.fully_connected(net3, 384, activation_fn=None, scope='fc3')
    print(net3.get_shape().as_list())
    net3 = tf.reshape(net3, [batch_size, 128,3])
    print(net3.get_shape().as_list())


    return net3, encoding


def get_loss(pred, label, goal_pcs, reg_weight=0.001):
    loss = tf.reduce_sum(tf.square(pred - goal_pcs))
    return loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        #print(outputs)
