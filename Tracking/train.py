import h5py
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Dense, Reshape, Flatten, Dropout
from keras.optimizers import SGD

import scipy.io as spio
import numpy as np

model = load_model('trained_vgg.h5')
for k in range(90):
    fName = './data/'+str(k+1)
    mat = spio.loadmat(fName)
    x_image = np.array(mat['x_image'])
    y_ht = np.array(mat['y_ht'])
    print str(k)+' '
    model.fit(x_image, y_ht, epochs=1, batch_size=100, verbose=1)
    # model.evaluate(x_image, y_ht, batch_size=100, verbose=1)

model.save('trained_vgg.h5');

