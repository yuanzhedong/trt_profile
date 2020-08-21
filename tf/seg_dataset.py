from PIL import Image
from glob import glob
import os
import numpy as np
import cv2

class SegmentationDataset():
    def __init__(self, size = 512, num_block = 4, block_size = 8):
        self.size = size
        # num of block per input
        self.num_block = num_block
        self.block_size = block_size


    def get_data(self, num_data):
        Xs = []
        Ys = []
        for i in range(num_data):
            X = np.zeros((self.size, self.size, 3))
            Y = np.zeros((self.size, self.size, 1))
            tls_x = np.random.randint(self.size, size=self.num_block)
            tls_y = np.random.randint(self.size, size=self.num_block)
            for i in range(self.num_block):
                tl_x = tls_x[i]
                tl_y = tls_y[i]
                min_x = tl_x
                max_x = min(self.size, tl_x + self.block_size)

                min_y = tl_y
                max_y = min(self.size, tl_y + self.block_size)

                X[min_x:max_x, min_y:max_y, :] = 1
                Y[min_x:max_x, min_y:max_y] = 1

            Xs.append(X)
            Ys.append(Y)
        return np.array(Xs), np.array(Ys)
