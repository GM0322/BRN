import numpy as np
from numpy import matlib


def Mask(in_width, out_width, channel, overlap=None):
    block_num = int(np.sqrt(channel))
    if overlap == None:
        overlap = (in_width * block_num - out_width) // (1 * (block_num - 1))
    mask = np.zeros((channel, out_width, out_width), dtype=np.float32)
    mask_one = np.ones((in_width, in_width), dtype=np.float32)
    left = matlib.repmat(np.array([1.0 / overlap * (i + 1) for i in range(overlap)], dtype=np.float32), in_width, 1)
    right = np.matlib.repmat(np.array([1 - 1.0 / overlap * (i) for i in range(overlap)], dtype=np.float32), in_width, 1)
    mask_one[0:overlap, :] = left.T
    mask_one[:, 0:overlap] = left
    mask_one[-overlap:, :] = right.T
    mask_one[:, -overlap:] = right
    corner1 = np.zeros((overlap, overlap), dtype=np.float32)
    for i in range(overlap):
        for j in range(overlap):
            if i > j:
                corner1[i, j] = 1.0 / overlap * (j + 1)
            else:
                corner1[i, j] = (i + 1) * 1.0 / overlap
    mask_one[0:overlap, 0:overlap] = corner1
    mask_one[0:overlap, -overlap:] = corner1[:, ::-1]
    mask_one[-overlap:, 0:overlap] = corner1[::-1, :]
    mask_one[-overlap:, -overlap:] = corner1[::-1, ::-1]
    for i in range(block_num):
        for j in range(block_num):
            mask[i * block_num + j, (in_width - overlap) * i:(i + 1) * in_width - i * overlap,
            (in_width - overlap) * j:(j + 1) * in_width - j * overlap] = mask_one

    mask = mask / mask.sum(axis=0)
    return mask

a = Mask(102, 512, 36)
a.tofile('../checkpoints/merge_mask.raw')
import matplotlib.pyplot as plt
for i in range(6):
    for j in range(6):
        plt.subplot(6,6,i*6+j+1)
        plt.imshow(a[i*6+j,:,:],cmap='gray')
plt.show()