from keras.losses import categorical_crossentropy
from keras.models import load_model
import numpy as np
from custom_datasets import wormdata
from custom_losses import w_categorical_crossentropy
from conf_mat import confusion_mat
data = wormdata(filepath='./data.mat', interval=10)
X_train, y_train = data.get_input(tag='train'), data.get_target(tag='train')
X_test, y_test = data.get_input(tag='test'), data.get_target(tag='test')
X_test = X_test.transpose((0,2,3,1))
X_train = X_train.transpose((0,2,3,1))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean_image = np.mean(X_train, axis=0)
X_test -= mean_image
X_test /= 128.
model = load_model('res18.h5', custom_objects={'w_categorical_crossentropy': w_categorical_crossentropy})
