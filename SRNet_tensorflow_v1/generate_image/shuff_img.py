import shutil

import os
from glob import glob
from random import shuffle

# Randomly chosen 4,000 images from BOSSbase and the entire BOWS2 dataset were used for training with 1,000 BOSSbase images set aside for validation.
# The remaining 5,000 BOSSbase images were used for testing.
# shuffle the dataset to the train.txt, valid.txt, test.txt

COVER_DIR =  "/data1/dataset/ALASKA256_v2/TIFF/"
STEGO_DIR =  "/data1/dataset/ALASKA256_v2/CPVS0_p50/"
train_cover = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/train_cover/"
train_stego = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/train_stego/"
valid_cover = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/valid_cover/"
valid_stego = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/valid_stego/"
test_cover = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/test_cover/"
test_stego = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.5p/test_stego/"
    
# train_cover_list = glob('C:\\Users\\Administrator\\Desktop\\train-BOSS-BOWS-256\\train_cover\\')
# train_stego_list = glob(STEGO_DIR + '/*.pgm')
# boss_cover_list = glob(COVER_DIR + '/Boss_256/*')
#boss_stego_list = glob(STEGO_DIR + '*.mat')
#boss_stego_list = glob(STEGO_DIR + '*.pgm')
boss_stego_list = glob(STEGO_DIR + '*.tif')
boss_stego_list = [a.replace(STEGO_DIR,'') for a in boss_stego_list]


print(len(boss_stego_list))

shuffle(boss_stego_list)
# 读取随机图片列表
#train_stego_list = boss_stego_list[0:54000]
#valid_stego_list = boss_stego_list[54000:60000]
#test_stego_list = boss_stego_list[60000:80000]


## 复制函数
def mycopy(srcpath,dstpath,filename):
    if not os.path.exists(srcpath):
        print("srcpath not exist!")
    if not os.path.exists(dstpath):
        print("dstpath not exist!")
    for root, dirs, files in os.walk(srcpath, True):
        if filename in files:
            # 如果存在就拷贝
            shutil.copy(os.path.join(root,filename),dstpath)
        else:
            # 不存在的话将文件信息打印出来
            print(filename)

# for filename in valid_stego_list:
#     mycopy(COVER_DIR, valid_cover, filename)
#     mycopy(STEGO_DIR, valid_stego, filename)
#     print(filename)
#
while len(train_cover + '/*') != 54000:
    print("len train:")
    print(len(glob(train_cover + '/*')))
    shuffle(boss_stego_list)
    train_stego_list = boss_stego_list[len(train_cover + '/*'):54000]
    for filename in train_stego_list:
        mycopy(COVER_DIR, train_cover, filename)
        mycopy(STEGO_DIR, train_stego, filename)
        print(filename)
        
# for filename in train_stego_list:
#     mycopy(COVER_DIR, train_cover, filename)
#     mycopy(STEGO_DIR, train_stego, filename)
#     print(filename)


shuffle(boss_stego_list)
test_stego_list = boss_stego_list[0:20000]
for filename in test_stego_list:
    mycopy(COVER_DIR, test_cover, filename)
    mycopy(STEGO_DIR, test_stego, filename)
    print(filename)

# save the list to the train.txt, valid.txt, test.txt
# train_txt = open('train.txt', 'w')
# for a in train_stego_list:
#    train_txt.write(a + "\n")
#train_txt.close()

#valid_txt = open('valid.txt', 'w')
#for a in valid_stego_list:
#    valid_txt.write(a + "\n")
#valid_txt.close()

#test_txt = open('test.txt', 'w')
#for a in test_stego_list:
#    test_txt.write(a + "\n")
#test_txt.close()

train_ds_size = len(train_stego_list) 
valid_ds_size = len(valid_stego_list)
print('train_ds_size: %i'%train_ds_size)
print('valid_ds_size: %i'%valid_ds_size)
# Testing 
test_ds_size = len(test_stego_list)
print('test_ds_size: %i'%test_ds_size)


