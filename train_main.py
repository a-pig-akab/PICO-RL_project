# By Shihang Wu and Weixiang Li(Code inherited from Weixuan Tang)

import imageio
import tensorflow as tf
# tf.disable_v2_behavior()
import numpy as np
import random
import argparse
import os
import scipy.io as sio
from batch_norm_layer import batch_norm_layer
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from SRNet_tensorflow_v1.SRNet import SRNet

# Parameter setting
parser = argparse.ArgumentParser(description="Set system parameters")
parser.add_argument('--gpu_num', type=str, default='2', help='set the gpu number')  # 添加一个x参数，默认值为1，类型为int
parser.add_argument('--train_img_path', type=str, default="./img/SZUBaseGray_256/", help='set train img path')
parser.add_argument('--test_img_path', type=str, default="./img/BOSS_256/", help='set test img path')
parser.add_argument('--iteration', type=int, default=200000, help='set the train iteration')
parser.add_argument('--use_img_of_train', type=float, default=1.0, help='set the percent of train img to use')
parser.add_argument('--test_payload', type=float, default=0.4, help='set the payload of test stego')
parser.add_argument('--use_tensorboard', type=str, default="true", help='set use the tensorboard to record the loss')
parser.add_argument('--save_path', type=str, default='./', help='set the path for model and test img')
parser.add_argument('--save_TesImg_iter', type=int, default=100, help='set iter to save the test img at one time')
parser.add_argument('--save_model_iter', type=int, default=100, help='set iter to save the model at one time')
parser.add_argument('--seed', type=int, default=1234, help='Sets the seed used to scramble the image')
parser.add_argument('--train_img_name', type=str, default="SZUBase", help='set train img name')
parser.add_argument('--test_img_name', type=str, default="BossBase", help='set test img name')
parser.add_argument('--star_iter', type=int, default=0, help='set star iter of train')
parser.add_argument('--load_model', type=str, default=None, help='set the load model(None is not load)')
parser.add_argument('--train_test', type=str, default='train', help='set the code is used for training or testing')
args = parser.parse_args()

# Correlation parameter reading
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
path1 = args.train_img_path  # path of training set
path2 = args.test_img_path

# Gets a list of names for training and test images
if args.train_test == 'train':
    fileList = []
    for (dirpath, dirnames, filenames) in os.walk(path1):
        fileList = filenames
    np.set_printoptions(threshold=10000000)
    random.seed(args.seed)
    random.shuffle(fileList)

fileList2 = []
for (dirpath2, dirnames2, filenames2) in os.walk(path2):
    fileList2 = filenames2

# # select the graphic card
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # PAYLOAD = 0.4  # Target embedding payload
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# path1 = "/data2/wushihang/wsh/img/SPAR-RL/SZUBaseGray_256/"  # path of training set
# # path1 = "/data2/wushihang/wsh/img/SPAR-RL/Alask_256_Final/"
# # path1 = "/data2/wushihang/liweixiang/SPAR-RL/img/SZUBaseGray_256/"  # path of training set
# path2 = "/data2/wushihang/liweixiang/SPAR-RL/img/BOSS_256/"    # path of testing set
# pathR = '/data2/wushihang/liweixiang/SPAR-RL/model/SZUBase/PICORL_HILL_p8/'
# if not os.path.exists(pathR):
#     os.makedirs(pathR)
#
# fileList = []
# for (dirpath, dirnames, filenames) in os.walk(path1):
#     fileList = filenames
# np.set_printoptions(threshold=10000000)
# random.seed(1234)
# random.shuffle(fileList)
#
# fileList2 = []
# for (dirpath2, dirnames2, filenames2) in os.walk(path2):
#     fileList2 = filenames2

# ******************************************* constant value settings ************************************************
img_1 = imageio.imread(path2 + '/' + fileList2[0])
NUM_ITERATION = args.iteration
# NUM_IMG = 10000  # The number of images used to train the network
if args.train_test == 'train':
    NUM_IMG = len(fileList)
USE_percent = args.use_img_of_train
BATCH_SIZE = 25
IMAGE_SIZE = img_1.shape[0]
NUM_CHANNEL = 1  # gray image
NUM_LABELS = 2  # binary classification
G_DIM = 16  # number of feature maps in generator
STRIDE = 2
KENEL_SIZE = 3
DKENEL_SIZE = 5
# PAYLOAD = 0.1  # Target embedding payload
PAD_SIZE = int((KENEL_SIZE - 1) / 2)
Initial_learning_rate = 0.0001
Adam_beta = 0.5
TANH_LAMBDA = 60  # To balance the embedding simulate and avoid gradient vanish problem

cover = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
is_training = tf.placeholder(tf.bool, name='is_training')  # True for training, false for test
PAYLOAD = tf.placeholder(tf.float32)  # randomly selected payload

# ********************************************* definition of the generator *********************************************************

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# -------------- contracting path ---------------------
with tf.variable_scope("Gen1") as scope:
    NUM = G_DIM * 1
    kernel1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM_CHANNEL, NUM], stddev=0.02),
                            name="kernel1_G")
    conv1_G = tf.nn.conv2d(cover / 255, kernel1_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv1_G")
    bn1_G = batch_norm_layer(conv1_G, is_training, 'bn1_G')
    # feature map shape: 128*128

with tf.variable_scope("Gen2") as scope:
    NUM = G_DIM * 2
    kernel2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM / 2), NUM], stddev=0.02),
                            name="kernel2_G")
    conv2_G = tf.nn.conv2d(lrelu(bn1_G, 0.2), kernel2_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv2_G")
    bn2_G = batch_norm_layer(conv2_G, is_training, 'bn2_G')
    # feature map shape: 64*64

with tf.variable_scope("Gen3") as scope:
    NUM = G_DIM * 4
    kernel3_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM / 2), NUM], stddev=0.02),
                            name="kernel3_G")
    conv3_G = tf.nn.conv2d(lrelu(bn2_G, 0.2), kernel3_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv3_G")
    bn3_G = batch_norm_layer(conv3_G, is_training, 'bn3_G')
    # feature map shape: 32*32

with tf.variable_scope("Gen4") as scope:
    NUM = G_DIM * 8
    kernel4_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM / 2), NUM], stddev=0.02),
                            name="kernel4_G")
    conv4_G = tf.nn.conv2d(lrelu(bn3_G, 0.2), kernel4_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv4_G")
    bn4_G = batch_norm_layer(conv4_G, is_training, 'bn4_G')
    #  feature map shape: 16*16

with tf.variable_scope("Gen5") as scope:
    NUM = G_DIM * 8
    kernel5_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel5_G")
    conv5_G = tf.nn.conv2d(lrelu(bn4_G, 0.2), kernel5_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv5_G")
    bn5_G = batch_norm_layer(conv5_G, is_training, 'bn5_G')
    # feature map shape: 8*8

with tf.variable_scope("Gen6") as scope:
    NUM = G_DIM * 8
    kernel6_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel6_G")
    conv6_G = tf.nn.conv2d(lrelu(bn5_G, 0.2), kernel6_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv6_G")
    bn6_G = batch_norm_layer(conv6_G, is_training, 'bn6_G')
    # feature map shape: 4*4

with tf.variable_scope("Gen7") as scope:
    NUM = G_DIM * 8
    kernel7_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel7_G")
    conv7_G = tf.nn.conv2d(lrelu(bn6_G, 0.2), kernel7_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv7_G")
    bn7_G = batch_norm_layer(conv7_G, is_training, 'bn7_G')
    # 2*2

with tf.variable_scope("Gen8") as scope:
    NUM = G_DIM * 8
    kernel8_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel8_G")
    conv8_G = tf.nn.conv2d(lrelu(bn7_G, 0.2), kernel8_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv8_G")
    bn8_G = batch_norm_layer(conv8_G, is_training, 'bn8_G')
    # 1*1

s = IMAGE_SIZE
s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64), int(
    s / 128)
# -------------- expanding path -----------------
with tf.variable_scope("Gen9") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s128, s128, NUM]
    kernel9_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel9_G")
    conv9_G = tf.nn.conv2d_transpose(tf.nn.relu(bn8_G), kernel9_G, out_shape, [1, STRIDE, STRIDE, 1], name="conv9_G")
    bn9_G = batch_norm_layer(conv9_G, is_training, 'bn9_G')
    bn9_G = tf.nn.dropout(bn9_G, 0.5)
    bn9_G = tf.concat([bn9_G, bn7_G], 3)

with tf.variable_scope("Gen10") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s64, s64, NUM]
    kernel10_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 2], stddev=0.02), name="kerne10_G")
    conv10_G = tf.nn.conv2d_transpose(tf.nn.relu(bn9_G), kernel10_G, out_shape, [1, STRIDE, STRIDE, 1], name="conv10_G")
    bn10_G = batch_norm_layer(conv10_G, is_training, 'bn10_G')
    bn10_G = tf.nn.dropout(bn10_G, 0.5)
    bn10_G = tf.concat([bn10_G, bn6_G], 3)

with tf.variable_scope("Gen11") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s32, s32, NUM]
    kernel11_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 2], stddev=0.02), name="kerne11_G")
    conv11_G = tf.nn.conv2d_transpose(tf.nn.relu(bn10_G), kernel11_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv11_G")
    bn11_G = batch_norm_layer(conv11_G, is_training, 'bn11_G')
    bn11_G = tf.nn.dropout(bn11_G, 0.5)
    bn11_G = tf.concat([bn11_G, bn5_G], 3)

with tf.variable_scope("Gen12") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s16, s16, NUM]
    kernel12_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 2], stddev=0.02), name="kerne12_G")
    conv12_G = tf.nn.conv2d_transpose(tf.nn.relu(bn11_G), kernel12_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv12_G")
    bn12_G = batch_norm_layer(conv12_G, is_training, 'bn12_G')
    bn12_G = tf.concat([bn12_G, bn4_G], 3)

with tf.variable_scope("Gen13") as scope:
    NUM = G_DIM * 4
    out_shape = [BATCH_SIZE, s8, s8, NUM]
    kernel13_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 4], stddev=0.02), name="kerne13_G")
    conv13_G = tf.nn.conv2d_transpose(tf.nn.relu(bn12_G), kernel13_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv13_G")
    bn13_G = batch_norm_layer(conv13_G, is_training, 'bn13_G')
    bn13_G = tf.concat([bn13_G, bn3_G], 3)

with tf.variable_scope("Gen14") as scope:
    NUM = G_DIM * 2
    out_shape = [BATCH_SIZE, s4, s4, NUM]
    kernel14_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 4], stddev=0.02), name="kerne14_G")
    conv14_G = tf.nn.conv2d_transpose(tf.nn.relu(bn13_G), kernel14_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv14_G")
    bn14_G = batch_norm_layer(conv14_G, is_training, 'bn14_G')
    bn14_G = tf.concat([bn14_G, bn2_G], 3)

with tf.variable_scope("Gen15") as scope:
    NUM = G_DIM
    out_shape = [BATCH_SIZE, s2, s2, NUM]
    kernel15_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM * 4], stddev=0.02), name="kerne15_G")
    conv15_G = tf.nn.conv2d_transpose(tf.nn.relu(bn14_G), kernel15_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv15_G")
    bn15_G = batch_norm_layer(conv15_G, is_training, 'bn15_G')
    bn15_G = tf.concat([bn15_G, bn1_G], 3)

with tf.variable_scope("Gen16") as scope:
    NUM = NUM_CHANNEL
    out_shape = [BATCH_SIZE, s, s, NUM]
    kernel16_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, G_DIM * 2], stddev=0.02),
                             name="kerne16_G")
    conv16_G = tf.nn.conv2d_transpose(tf.nn.relu(bn15_G), kernel16_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv16_G")

# Embeding_prob = tf.nn.relu(tf.nn.sigmoid(conv16_G) - 0.5)
# Embeding_prob = tf.nn.relu(tf.nn.sigmoid(conv16_G) - 1 / 3)
# Embeding_prob = tf.nn.relu(tf.nn.sigmoid(conv16_G) / 1.5)
# rho = tf.nn.relu(tf.nn.sigmoid(conv16_G) - 0.5)
# rho = tf.nn.relu(tf.nn.sigmoid(conv16_G))
rho = tf.nn.sigmoid(conv16_G)
# Lambda = 40
# Lambda = 128.9 * tf.pow(PAYLOAD, -0.2069) - 116.3
# Lambda = 98.62 * tf.pow(PAYLOAD, -0.251) - 84.12    # BOSSBase-100
# Lambda = 121.9 * tf.pow(PAYLOAD, -0.2124) - 108    # BOSSBase-10000
# Lambda = 101.4 * tf.pow(PAYLOAD, -0.2609) - 88.61    # SZUBase-all(41314)
# Lambda = 100.3 * tf.pow(PAYLOAD, -0.2591) - 87.05    # SZUBase-1000
# Lambda = -114.8968 * tf.pow(PAYLOAD, 0.1192) + 132.0939    # SZUBase-1000-MiPOD-p8
Lambda = 149.5766 * tf.pow(PAYLOAD, -0.2163) - 137.4412    # SZUBase-1000-HILL-p8

# Lambda_converted = tf.reshape(
#     tf.broadcast_to(Lambda, [rho.shape[0], rho.shape[1], rho.shape[2], rho.shape[3]]), tf.shape(rho))
# prob = (tf.exp(-Lambda_converted*rho))/(1+2*tf.exp(-Lambda_converted*rho))

prob = (tf.exp(-Lambda*rho))/(1+2*tf.exp(-Lambda*rho))
# prob = (tf.exp(-tf.multiply(rho,Lambda)))/(1+2*tf.exp(-tf.multiply(rho,Lambda)))
# rhoP1 = rho
# rhoM1 = rho
proChangeP = prob
proChangeM = prob
# proChangeP = (tf.exp(-Lambda*rhoP1))/(1+tf.exp(-Lambda*rhoP1)+tf.exp(-Lambda*rhoM1))
# proChangeM = (tf.exp(-Lambda*rhoM1))/(1+tf.exp(-Lambda*rhoP1)+tf.exp(-Lambda*rhoM1))

Embeding_prob_shape = rho.get_shape().as_list()
output = rho

# ***************************************************  double-tanh function for embedding simulation ***************************************************

# proChangeP = Embeding_prob / 2.0
# proChangeM = Embeding_prob / 2.0
# Embeding_prob_shape = Embeding_prob.get_shape().as_list()
noise = tf.placeholder(tf.float32, Embeding_prob_shape)  # noise holder

modification_0 = tf.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
modification_p1 = tf.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
modification_m1 = -1 * tf.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
modification_temp_equal = tf.where(noise < proChangeM, modification_m1, modification_0)
modification_equal = tf.where(noise > 1 - proChangeP, modification_p1, modification_temp_equal)
modification = modification_equal
stego = cover + modification_equal

# *************************************************** definition of the discriminator **************************************************************

Img = tf.concat([cover, stego], 0)
y_array = np.zeros([BATCH_SIZE * 2, NUM_LABELS], dtype=np.float32)
for i in range(0, BATCH_SIZE):
    y_array[i, 1] = 1
for i in range(BATCH_SIZE, BATCH_SIZE * 2):
    y_array[i, 0] = 1
y = tf.constant(y_array)

Img_label = tf.constant(y_array)

# *********************** SRNet model ***********************
srnet = SRNet(is_training=True)

D_y = srnet._build_model(Img)

correct_predictionS = tf.equal(tf.argmax(D_y, 1), tf.argmax(Img_label, 1))
accuracyD = tf.reduce_mean(tf.cast(correct_predictionS, tf.float32))
lossD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_y, labels=Img_label))  # loss of D

y_ = D_y
y = Img_label
y_Cover, y_Stego = tf.split(y_, 2, axis=0)
yCover, yStego = tf.split(y, 2, axis=0)

lossCover = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_Cover, labels=yCover))
lossStego = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_Stego, labels=yStego))
lossM = tf.reduce_mean(tf.abs(modification))

# ******************************************************* loss function ************************************************************************

gamma = 1
# lambda_ent = 1e-7
lambda_ent = 100
# proChangeP = Embeding_prob / 2.0 + 1e-5
# proChangeM = Embeding_prob / 2.0 + 1e-5
proUnchange = 1 - proChangeP - proChangeM - 1e-8
proChangeM1 = proChangeM + 1e-8
proChangeP1 = proChangeP + 1e-8
entropy = tf.reduce_sum(
    -(proChangeP1) * tf.log(proChangeP1) / tf.log(2.0) - (proChangeM1) * tf.log(proChangeM1) / tf.log(
        2.0) - proUnchange * tf.log(proUnchange) / tf.log(2.0), reduction_indices=[1, 2, 3])
# Payload_learned = tf.reduce_sum(entropy, reduction_indices=0) / IMAGE_SIZE / IMAGE_SIZE / BATCH_SIZE
Payload_learned = entropy / IMAGE_SIZE / IMAGE_SIZE

# Capacity = IMAGE_SIZE * IMAGE_SIZE * PAYLOAD
# lossEntropy = tf.reduce_mean(tf.pow(entropy - Capacity, 2), reduction_indices=0)
lossEntropy = tf.reduce_mean(tf.pow(Payload_learned - PAYLOAD, 2), reduction_indices=0)

# -------------------loss of the generator -------------

gradient_equal = tf.multiply(tf.gradients(lossD, modification_equal), 1e7)
reward_equal = tf.multiply(gradient_equal, modification_equal)
mask_add_equal = tf.where(tf.equal(modification_equal, 1), tf.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]),
                          tf.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]))
mask_minus_equal = tf.where(tf.equal(modification_equal, -1),
                            tf.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]),
                            tf.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL]))
target_equal = tf.multiply(
    tf.multiply(mask_add_equal, tf.log(proChangeP1)) + tf.multiply(mask_minus_equal, tf.log(proChangeM1)), reward_equal)
lossRL = -tf.reduce_mean(tf.reshape(target_equal, [BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNEL, 1]))
lossGen = lambda_ent * lossEntropy + gamma * lossRL

# -------------------trainable variables----------------

variables = tf.trainable_variables()
paramsG = [v for v in variables if (v.name.startswith('Gen'))]
# paramsD = [v for v in variables if (v.name.startswith('Group'))]
paramsD = [v for v in variables if (v.name.startswith('Layer'))]
# -------------------- optimizers ---------------------------

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optGen = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossGen, var_list=paramsG)
    optDis = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossD, var_list=paramsD)

global_variables = tf.global_variables()

# **************************************************************** adversary training process ***************************************************************************


if args.train_test == 'train':
    use_img_num = int(NUM_IMG * USE_percent) - 1
    # save fold name
    name = f'{args.train_img_name}_PICORL_original'

    # save path
    pathR = args.save_path + name + '/'

    # create fold
    if not os.path.exists(pathR):
        os.makedirs(pathR)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        # whether load model or not
        if args.load_model != None:
            saver.restore(sess, args.load_model)

        data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
        count = 0

        if args.use_tensorboard == 'true':
            writer = SummaryWriter(f'runs/{name}')

        # the train loop
        for iteration in range(args.star_iter, NUM_ITERATION + 1):
            for j in range(BATCH_SIZE):
                count = count % use_img_num
                imc = imageio.imread(path1 + '/' + fileList[count])
                data_x[j, :, :, 0] = imc
                count = count + 1

            data_noise = np.random.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
            data_payload = np.random.rand() * 0.75 + 0.05
            _, c1, c2, c3, c4, c5, c6, c7, c8, c9 = sess.run([optDis, lossEntropy, lossCover, lossStego, lossM, rho, entropy, lossD, prob, Payload_learned],
                                         feed_dict={cover: data_x, noise: data_noise, PAYLOAD: data_payload, is_training: True})
            # print("D", epoch, c1, c2, c3, c4)
            print("Dis: ", ' epoch: %d' % iteration, ' lossE: %f' % c1, ' lossC: %f' % c2, ' lossS: %f' % c3, ' lossM: %f' % c4, ' lossD: %f' % c7, ' payload_selected: %f' % data_payload, ' payload_learned: %f' % c9[0])
            c51 = c5[0, :, :, 0]

            _, c1, c2, c3, c4, c5, c6, c7, c8, c9 = sess.run([optGen, lossEntropy, lossCover, lossStego, lossM, rho, entropy, lossGen, prob, Payload_learned],
                                         feed_dict={cover: data_x, noise: data_noise, PAYLOAD: data_payload, is_training: True})
            # print("G", epoch, c1, c2, c3, c4)
            print("Gen: ", ' epoch: %d' % iteration, ' lossE: %f' % c1, ' lossC: %f' % c2, ' lossS: %f' % c3, ' lossM: %f' % c4, ' lossG: %f' % c7, ' payload_selected: %f' % data_payload, ' payload_learned: %f' % c9[0])
            c52 = c5[0, :, :, 0]
            c5diff_rho = c52 - c51

            # use tensorboard to record the loss of train
            if args.use_tensorboard == 'true':
                writer.add_scalar('lossEntropy', c1, global_step=iteration + 1)
                writer.add_scalar('lossCover', c2, global_step=iteration + 1)
                writer.add_scalar('lossStego', c3, global_step=iteration + 1)
                writer.add_scalar('lossM', c4, global_step=iteration + 1)
                # writer.add_scalar('lossGen', c5, global_step=iteration + 1)

            # save train model
            if iteration > 0 and iteration % args.save_model_iter == 0:
                saver = tf.train.Saver()
                saver.save(sess, pathR + 'Gan' + '%d' % iteration + '.ckpt')

            # save test img library
            if iteration > 0 and iteration % args.save_TesImg_iter == 0:
                pathF = pathR + 'Gan' + str(iteration)
                if not os.path.exists(pathF):
                    os.makedirs(pathF)

                count1 = 0
                pbar = tqdm(total=len(fileList2), desc='Saving the test img library')
                while count1 < len(fileList2):
                    pbar.update(BATCH_SIZE)
                    for j in range(BATCH_SIZE):
                        if not (count1 < len(fileList2)):
                            break
                        imc = imageio.imread(path2 + '/' + fileList2[count1])
                        data_x[j, :, :, 0] = imc
                        count1 = count1 + 1
                    data_noise = np.random.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
                    data_payload = args.test_payload
                    Rho, Prob, Cover, Stego = sess.run([output, prob, cover, stego],
                                                         feed_dict={cover: data_x, noise: data_noise, PAYLOAD: data_payload, is_training: False})
                    for j in range(BATCH_SIZE):
                        TheCover = Cover[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                        TheStego = Stego[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                        TheRho = Rho[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                        TheProbability = Prob[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                        sio.savemat(pathF + '/' + fileList2[count1 - BATCH_SIZE + j][0:-4] + '.mat',
                                    {'stego': TheStego, 'cover': TheCover, 'rho': TheRho, 'probability': TheProbability})


# ************************* adversary testing process ***********************************************
elif args.train_test == 'test':
    # save fold name
    name = f'{args.test_img_name}_PICORL_original_{args.test_payload}_onlyTest'

    # save path
    pathR = args.save_path + name + '/'

    # create fold
    if not os.path.exists(pathR):
        os.makedirs(pathR)

    # create sess and begin test
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        saver.restore(sess, args.load_model)

        data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])

        # save test img library

        pathF = pathR + 'Gan_TesImg'
        if not os.path.exists(pathF):
            os.makedirs(pathF)

        count1 = 0
        pbar = tqdm(total=len(fileList2), desc='Saving the test img library')
        while count1 < len(fileList2):
            pbar.update(BATCH_SIZE)
            for j in range(BATCH_SIZE):
                if not (count1 < len(fileList2)):
                    break
                imc = imageio.imread(path2 + '/' + fileList2[count1])
                data_x[j, :, :, 0] = imc
                count1 = count1 + 1
            data_noise = np.random.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
            data_payload = args.test_payload
            Rho, Prob, Cover, Stego = sess.run([output, prob, cover, stego],
                                               feed_dict={cover: data_x, noise: data_noise, PAYLOAD: data_payload,
                                                          is_training: False})
            for j in range(BATCH_SIZE):
                TheCover = Cover[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                TheStego = Stego[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                TheRho = Rho[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                TheProbability = Prob[j, :, :, :].reshape([IMAGE_SIZE, IMAGE_SIZE])
                sio.savemat(pathF + '/' + fileList2[count1 - BATCH_SIZE + j][0:-4] + '.mat',
                            {'stego': TheStego, 'cover': TheCover, 'rho': TheRho, 'probability': TheProbability})

