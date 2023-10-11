import numpy as np
from scipy import misc, io
import imageio
from glob import glob
import random
from random import random as rand
from random import shuffle

def gen_flip_and_rot(cover_dir, stego_dir, sample_file, thread_idx, n_threads):
    cover_list_all = sorted(glob(cover_dir + '/*'))
    stego_list_all = sorted(glob(stego_dir + '/*'))
    with open(sample_file) as f:
        img_lines = f.readlines()
        cover_list = [(cover_dir + a.replace('\n', '')) for a in img_lines]
        stego_list = [(stego_dir + a.replace('\n', '')) for a in img_lines]
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')                         # 判断是jpeg还是空域
    if load_mat:
        img = io.loadmat(cover_list[0])['im']                       # 读jpeg数据
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')      # batch放一项放cover 一项放stego 一整张图
    else:
        img = imageio.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    
    iterable = list(zip(cover_list, stego_list))
    while True:
        shuffle(iterable)
        for cover_path, stego_path in iterable:
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['im']
                batch[1,:,:,0] = io.loadmat(stego_path)['im']
            else:
                batch[0,:,:,0] = imageio.imread(cover_path)
                batch[1,:,:,0] = imageio.imread(stego_path)
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]        # yield类似return
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]
                              

def gen_valid(cover_dir, stego_dir, sample_file, thread_idx, n_threads):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    with open(sample_file) as f:
        img_lines = f.readlines()
        cover_list = [cover_dir + a.replace('\n', '') for a in img_lines]
        stego_list = [stego_dir + a.replace('\n', '') for a in img_lines]
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['im']
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = imageio.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    img_shape = img.shape
    
    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['im']
                batch[1,:,:,0] = io.loadmat(stego_path)['im']
            else:
                batch[0,:,:,0] = imageio.imread(cover_path)
                batch[1,:,:,0] = imageio.imread(stego_path)
            yield [batch, labels]
