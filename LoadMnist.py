# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:02:13 2017

@author: aumale
"""

class LoadMNist:
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def load(self):
        img_train, label_train = self._load("training")
        img_test, label_test = self._load("testing")
        return img_train, label_train, img_test, label_test

    def _load(self, dataset, digits=np.arange(10)):
        if dataset == 'training':
            fdata = os.path.join(self.rootdir, 'train-images.idx3-ubyte')
            flabel = os.path.join(self.rootdir, 'train-labels.idx1-ubyte')
        elif dataset == 'testing':
            fdata = os.path.join(self.rootdir, 't10k-images.idx3-ubyte')
            flabel = os.path.join(self.rootdir, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError('dataset must be traning or testing')

        with open(fdata, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            print(magic, num, rows, cols )
            
            img = np.fromfile(f, dtype=np.uint8).reshape((num, rows * cols))
            print(img.shape)
            img = img.ravel() * 1.0 / 255
            img = img.reshape((num, rows * cols))
            print(img.shape)

        with open(flabel, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            label = np.fromfile(f, dtype=np.uint8)

        return img, label
