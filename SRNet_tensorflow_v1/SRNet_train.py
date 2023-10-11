# %%

import os
import sys

# from tflib.generator import gen_flip_and_rot, gen_valid
# from tflib.utils_multistep_lr import AdamaxOptimizer, train, test_dataset

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # set a GPU (with GPU Number)
home = os.path.expanduser("~")
sys.path.append(home + '/tflib/')  # path for 'tflib' folder
from SRNet import *

# %%

train_batch_size = 32
valid_batch_size = 40
max_iter = 500000
train_interval = 100
valid_interval = 5000
save_interval = 5000
num_runner_threads = 10

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and
# save them as '.mat' files with variable name "im". Put the '.mat' files in thier corresponding directoroies. Make sure
# all mat files in the directories can be loaded in Python without any errors.

cover_dir = '/data2/liuxulong/dataset/BOSSBase-224/'
stego_dir = '/data2/liuxulong/dataset/Stego/BOSSBase-224-HILL-0.4/'
train_file = '/data2/liuxulong/dataset/config/BOSSBase_train.txt'
valid_file = '/data2/liuxulong/dataset/config/BOSSBase_valid.txt'
test_file = '/data2/liuxulong/dataset/config/BOSSBase_test.txt'

# TRAIN_COVER_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/train_cover/"
# TRAIN_STEGO_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/train_stego/"
#
# VALID_COVER_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/valid_cover/"
# VALID_STEGO_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/valid_stego/"

# train_gen = partial(gen_flip_and_rot,
#                     TRAIN_COVER_DIR, TRAIN_STEGO_DIR)
# valid_gen = partial(gen_valid,
#                     VALID_COVER_DIR, VALID_STEGO_DIR)
train_gen = partial(gen_flip_and_rot,
                    cover_dir, stego_dir, train_file)
valid_gen = partial(gen_valid,
                    cover_dir, stego_dir, valid_file)

LOG_DIR = "/data/liuxulong/PSte-CL/results/SRNet/LogFile/" # path for a log direcotry
#load_path = LOG_DIR + 'Model_500000.ckpt'  # continue training from a specific checkpoint
load_path = None  # training from scratch

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

with open(train_file) as f:
    train_list = f.readlines()
    f.close()
with open(valid_file) as f:
    valid_list = f.readlines()
    f.close()
# train_ds_size = len(glob(TRAIN_COVER_DIR + '/*')) * 2
# valid_ds_size = len(glob(VALID_COVER_DIR + '/*')) * 2
train_ds_size = len(train_list) * 2
valid_ds_size = len(valid_list) * 2
print('train_ds_size: %i' % train_ds_size)
print('valid_ds_size: %i' % valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")

optimizer = AdamaxOptimizer
boundaries = [400000]  # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates

train(SRNet, train_gen, valid_gen, train_batch_size, valid_batch_size, valid_ds_size,\
      optimizer, boundaries, values, train_interval, valid_interval, max_iter,\
      save_interval, LOG_DIR, num_runner_threads, load_path)
