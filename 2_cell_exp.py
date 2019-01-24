import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import scipy.io
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='2_pointAE_simple_1024', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=30000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=3, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

mat = scipy.io.loadmat('agglos.mat')

pc3nr1 = mat['data'][0]
pc3nr2 = mat['data'][1]
pc3nr3 = mat['data'][2]
pc4nr1 = mat['data'][3]
pc4nr2 = mat['data'][4]
pc4nr3 = mat['data'][5]

def transform_to_base(pc):
    minX = min(pc,key=lambda item:item[0])[0]
    minY = min(pc,key=lambda item:item[1])[1]
    minZ = min(pc,key=lambda item:item[2])[2]
    for i in range(len(pc)):
        pc[i][0] = pc[i][0]-minX
        pc[i][1] = pc[i][1]-minY
        pc[i][2] = pc[i][2]-minZ 

transform_to_base(pc3nr1)
transform_to_base(pc3nr2)
transform_to_base(pc3nr3)
transform_to_base(pc4nr1)
transform_to_base(pc4nr2)
transform_to_base(pc4nr3)
pc3nr1 = sorted(pc3nr1, key=lambda x: x[0])
pc3nr2 = sorted(pc3nr2, key=lambda x: x[0])
pc3nr3 = sorted(pc3nr3, key=lambda x: x[0])
pc4nr1 = sorted(pc4nr1, key=lambda x: x[0])
pc4nr2 = sorted(pc4nr2, key=lambda x: x[0])
pc4nr3 = sorted(pc4nr3, key=lambda x: x[0])



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def plotPC(pc):
    fig = pyplot.figure()
    ax = Axes3D(fig)


    x = []
    y = []
    z = []


    for i in range(len(pc)):
      x.append(pc[i,0])
      y.append(pc[i,1])
      z.append(pc[i,2])

    ax.scatter(x, y, z)
    pyplot.show()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.0001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, goal_pcs, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, enc = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, goal_pcs)
            tf.summary.scalar('loss', loss)

            #correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            #tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'goal_pcs': goal_pcs, 
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'enc': enc, 
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            #eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 100 == 0:
                my_eval_one_epoch(sess, ops, train_writer)
                #save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                #log_string("Model saved in file: %s" % save_path)


trainEnc = [[],[],[]]
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    #np.random.shuffle(train_file_idxs)
    
    for fn in range(1):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[3]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss = 0
       
        for batch_idx in range(1):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            orig_pcs = current_data[start_idx:end_idx, :, :]
            orig_pcs = np.asarray([pc3nr1, pc3nr2, pc3nr3])#, pc4nr1, pc4nr2, pc4nr3])
            # Augment batched point clouds by rotation and jittering
            #rotated_data = provider.rotate_point_cloud(orig_pcs)
            jittered_data = provider.jitter_point_cloud(orig_pcs)
            used_pcs = jittered_data
            #plotPC(used_pcs[1])
            feed_dict = {ops['pointclouds_pl']: used_pcs,
                         ops['goal_pcs']: orig_pcs,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val, encoding = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred'], ops['enc']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            #pred_val = tf.convert_to_tensor(pred_val)
            #print(pred_val.get_shape().as_list())
            
            #used_pcs_tensor = tf.convert_to_tensor(used_pcs)
            #print(used_pcs_tensor.get_shape().as_list())
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor)))
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor))) 
            loss = np.sum(np.square(np.subtract(pred_val, orig_pcs)))           
            #print(loss.get_shape().as_list())
            #tf.Print(loss, [loss])
            trainEnc[0] = encoding[0]
            trainEnc[1] = encoding[1]
            trainEnc[2] = encoding[2]
        log_string('mean loss: %f32' % loss)

        




def my_eval_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    #np.random.shuffle(train_file_idxs)
    
    for fn in range(1):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[3]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss = 0
       
        for batch_idx in range(1):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            orig_pcs = current_data[start_idx:end_idx, :, :]
            orig_pcs = np.asarray([pc4nr1, pc4nr2, pc4nr3])
            # Augment batched point clouds by rotation and jittering
            #rotated_data = provider.rotate_point_cloud(orig_pcs)
            jittered_data = provider.jitter_point_cloud(orig_pcs)
            used_pcs = jittered_data
            #plotPC(used_pcs[1])
            feed_dict = {ops['pointclouds_pl']: used_pcs,
                         ops['goal_pcs']: orig_pcs,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val, encoding = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred'], ops['enc']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            #pred_val = tf.convert_to_tensor(pred_val)
            #print(pred_val.get_shape().as_list())
            
            #used_pcs_tensor = tf.convert_to_tensor(used_pcs)
            #print(used_pcs_tensor.get_shape().as_list())
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor)))
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor)))
            print(trainEnc)
            print('distance from first:')
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[0], encoding[0])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[0], encoding[1])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[0], encoding[2])))))
            print('distance from second:')
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[1], encoding[0])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[1], encoding[1])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[1], encoding[2])))))
            print('distance from third:')
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[2], encoding[0])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[2], encoding[1])))))
            print(np.sqrt(np.sum(np.square(np.subtract(trainEnc[2], encoding[2])))))            
            plotPC(used_pcs[0])
            plotPC(pred_val[0])
            plotPC(used_pcs[1])
            plotPC(pred_val[1]) 
            loss = np.sum(np.square(np.subtract(pred_val, orig_pcs)))           
            #print(loss.get_shape().as_list())
            #tf.Print(loss, [loss])
        log_string('mean loss: %f32' % loss)

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    #np.random.shuffle(train_file_idxs)
    
    for fn in range(1):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[3]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss = 0
       
        for batch_idx in range(1):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            orig_pcs = current_data[start_idx:end_idx, :, :]
            orig_pcs = np.asarray([pc3nr1, pc3nr2])
            # Augment batched point clouds by rotation and jittering
            #rotated_data = provider.rotate_point_cloud(orig_pcs)
            jittered_data = provider.jitter_point_cloud(orig_pcs)
            used_pcs = jittered_data

            feed_dict = {ops['pointclouds_pl']: orig_pcs,
                         ops['goal_pcs']: orig_pcs,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            #pred_val = tf.convert_to_tensor(pred_val)
            #print(pred_val.get_shape().as_list())
            
            #used_pcs_tensor = tf.convert_to_tensor(used_pcs)
            #print(used_pcs_tensor.get_shape().as_list())
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor)))
            #loss = tf.reduce_sum(tf.square(tf.subtract(pred_val, used_pcs_tensor))) 
            plotPC(used_pcs[0])
            plotPC(pred_val[0])
            plotPC(used_pcs[1])
            plotPC(pred_val[1])
            loss = np.sum(np.square(np.subtract(pred_val, orig_pcs)))           
            #print(loss.get_shape().as_list())
            #tf.Print(loss, [loss])
        log_string('mean loss: %f32' % loss)    


if __name__ == "__main__":
    
    train()
    LOG_FOUT.close()
