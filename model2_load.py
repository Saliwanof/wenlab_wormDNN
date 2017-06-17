import h5py
from keras.models import Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Dense, Reshape, Flatten, Dropout, RepeatVector
from keras.layers import Input, Embedding
from keras.optimizers import Adam

weights_path = 'vgg16_weights.h5'
img_width, img_height = 256,192

image = Input(shape=(img_width*img_height,), name='main_input')
posinfo = Input(shape=(1,))

mlp1_output = Dense(64, activation='relu')(posinfo)

x = Reshape((256,192))(image)
x = RepeatVector(3)(x)

x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', data_format='channels_first')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', data_format='channels_first')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', data_format='channels_first')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', data_format='channels_first')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', data_format='channels_first')(x)
x = ZeroPadding2D((1, 1), data_format='channels_first')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', data_format='channels_first')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

x = Flatten()(x)

x = keras.layers.concatenate([x, mlp1_output])

x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)

main_output = Dense(5, activation='softmax', name='main_output')(x)

model = Model(inputs=main_input, outputs=main_output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

# model.add(Dense(4096, activation='relu', name='dense1'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu', name='dense2'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax', name='dense3'))

# model.load_weights(weights_path,by_name=True)

# model.compile(loss='categorical_crossentropy',
              # optimizer='Adam',
              # metrics=['accuracy'])

# get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])

mat = spio.loadmat('1.mat')
video = np.array(mat.get('video'))
video = np.reshape(video,(43494,192,256,1))
position = np.array(mat.get('headslice'))

model.fit(video[:1000,:,:,:], position[:1000,:,:], epochs=3, batch_size=1, verbose=1)