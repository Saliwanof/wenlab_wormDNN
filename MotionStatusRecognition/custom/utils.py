from sklearn.metrics import confusion_matrix
import h5py
from keras.models import load_model
import numpy as np

def confusion_mat(model, X_input, Y_true):
    Y_pred = model.predict(X_input, batch_size=32, verbose=1)
    Y_pred = classes_to_labels(probas_to_classes(Y_pred))
    Y_true = classes_to_labels(Y_true)
    conf_mat = confusion_matrix(Y_true, Y_pred)
    print(conf_mat)

def probas_to_classes(probas):
    nb_samples = probas.shape[0]
    nb_classes = probas.shape[1]
    max_proba = np.amax(probas, axis=1)
    for i in range(nb_classes):
        probas[:,i] = np.equal(probas[:,i],max_proba)
    return probas

def classes_to_labels(classes):
    nb_samples = classes.shape[0]
    nb_classes = classes.shape[1]
    labels = np.zeros(nb_samples)
    for i in range(nb_classes):
        labels += (classes[:,i] == 1) * (i+1)
    return labels

#
