import h5py
import numpy as np

class wormdata(object):
    def __init__(self, filepath='./data.mat', interval=10):
        frame_start = 0
        frame_end = 100000
        with h5py.File(filepath) as f:
            x_img = np.array(f['x_img']) # (100000,224,224)
            self.y_stat = np.array(f['y_stat'])[frame_start+interval:frame_end-interval,:] # (100000-2*10,5)
        frame_first = x_img[frame_start:frame_end-interval*2,:,:].reshape([-1, 224*224])
        frame_second = x_img[frame_start+interval:frame_end-interval,:,:].reshape([-1, 224*224])
        frame_third = x_img[frame_start+2*interval:frame_end,:,:].reshape([-1, 224*224])
        self.seq = np.concatenate((frame_first, frame_second, frame_third), axis=1).reshape([-1, 3, 224, 224])
        self.nb_samples = self.y_stat.shape[0]
        
    def get_target(self, tag='train'):
        if tag == 'train':
            return self.y_stat[:int(self.nb_samples * .9), :]
        else:
            return self.y_stat[int(self.nb_samples * .9):, :]
        
    def get_input(self, tag='train'):
        if tag == 'train':
            return self.seq[:int(self.nb_samples * .9), :, :, :]
        else:
            return self.seq[int(self.nb_samples * .9):, :, :, :]
        
