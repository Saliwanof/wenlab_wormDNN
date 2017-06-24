import h5py
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Dense, Reshape, Flatten, Dropout, RepeatVector
from keras.optimizers import Adam

import scipy.io as spio
import numpy as np

weights_path = 'vgg16_weights.h5'
img_width, img_height = 50, 50

model = Sequential()

model.add(Flatten(input_shape=(50,50)))
model.add(RepeatVector(3))
model.add(Reshape((3,50,50)))

model.add(ZeroPadding2D((1, 1), data_format='channels_first'))

model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2', data_format='channels_first'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2', data_format='channels_first'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3', data_format='channels_first'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3', data_format='channels_first'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2', data_format='channels_first'))
model.add(ZeroPadding2D((1, 1), data_format='channels_first'))
model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3', data_format='channels_first'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

model.add(Flatten())
model.add(Dense(4096, activation='relu', name='dense1_'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='dense2_'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax', name='dense3_'))

model.load_weights(weights_path,by_name=True)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

# get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])

for k in range(90):
    fName = './data/'+str(k+1)
    mat = spio.loadmat(fName)
    x_image = np.array(mat['x_image'])
    y_ht = np.array(mat['y_ht'])

    model.fit(x_image, y_ht, epochs=1, batch_size=100, verbose=1)

model.save('trained_vgg.h5');
