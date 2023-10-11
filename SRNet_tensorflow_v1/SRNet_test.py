import os
import sys

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

# %%
print('******************')
# Testing
# Cover and Stego directories for testing
# TEST_COVER_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/test_cover/"
# TEST_STEGO_DIR = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/test_stego/"
cover_dir = '/data2/liuxulong/dataset/BOSSBase-224/'
stego_dir = '/data2/liuxulong/dataset/Stego/BOSSBase-224-HILL-0.4/'
train_file = '/data2/liuxulong/dataset/config/BOSSBase_train.txt'
valid_file = '/data2/liuxulong/dataset/config/BOSSBase_valid.txt'
test_file = '/data2/liuxulong/dataset/config/BOSSBase_test.txt'

test_batch_size = 40
LOG_DIR = "/data/liuxulong/PSte-CL/results/SRNet/LogFile/"
LOAD_CKPT = LOG_DIR + 'Model_500000.ckpt'  # loading from a specific checkpoint

test_gen = partial(gen_valid, \
                   cover_dir, stego_dir, test_file)
with open(test_file) as f:
    test_list = f.readlines()
    f.close()
test_ds_size = len(test_list) * 2
print('test_ds_size: %i' % test_ds_size)

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

test_dataset(SRNet, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)