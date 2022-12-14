import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from pickle import dump, load
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from os.path import exists


def data_name(path, val_split):
    name = path.split('/')
    name = name[-1].split('.')
    return name[0] + 'Valid{}%.hdf5'.format(int(val_split * 100))


def generate_normalized_dataset(path, name, new_path, n_train, std_path, minmax_path):
    f = h5py.File(path, "r")
    x = np.array(f[name])
    f.close()

    train = x[:n_train]
    valid = x[n_train:]

    std_scaler = TimeSeriesScalerMeanVariance()
    minmax_scaler = TimeSeriesScalerMinMax()

    ts_train = minmax_scaler.fit_transform(std_scaler.fit_transform(train))
    ts_val = minmax_scaler.transform(std_scaler.transform(valid))

    dump(std_scaler, open(std_path, 'wb'))
    dump(minmax_scaler, open(minmax_path, 'wb'))

    nf = h5py.File(new_path, "w")
    nf.create_dataset(name, data=np.concatenate((ts_train, ts_val), axis=0))
    nf.close()


class ECGSequence(Sequence):

    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02, preprocessing=False):
        namefile = data_name(path_to_hdf5, val_split)
        path = './trainData/preprocessed/{}'.format(namefile)
        std_scaler_path = './scalers/stdScaler{}%.pkl'.format(int(val_split*100))
        minmax_scaler_path = './scalers/minMaxScaler{}%.pkl'.format(int(val_split*100))
        n_samples = len(pd.read_csv(path_to_csv, header=None))
        n_train = math.ceil(n_samples * (1 - val_split))

        if preprocessing:
            print('using preprocessed data')
            if not exists(path) or not exists(std_scaler_path) or not exists(minmax_scaler_path):
                generate_normalized_dataset(path_to_hdf5, hdf5_dset, path, n_train, std_scaler_path, minmax_scaler_path)

        train_seq = cls(path, hdf5_dset, val_split, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path, hdf5_dset, val_split, path_to_csv, batch_size, start_idx=n_train)

        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, val_split, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None, is_test=False, preprocessing=False):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv, header=None).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        if is_test and preprocessing:
            print('using preprocessed test set')
            std_scaler = load(open('./scalers/stdScaler{}%.pkl'.format(int(val_split*100)), 'rb'))
            minmax_scaler = load(open('./scalers/minMaxScaler{}%.pkl'.format(int(val_split*100)), 'rb'))
            self.x = minmax_scaler.transform(std_scaler.transform(self.x))
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
