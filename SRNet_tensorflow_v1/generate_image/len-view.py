from glob import glob

test_stego = "/data/liuxulong/SRNet-C/ALASKA256-v2-C-0.3p/train_stego"
size = len(glob(test_stego + '/*'))
print(size)