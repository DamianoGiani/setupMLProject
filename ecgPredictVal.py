import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from ecgDatasets import ECGSequence
import math
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to pickle file containing tracings')
    parser.add_argument('path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='percentual used of the training set as validation')
    parser.add_argument('--output_file', default="./outputs/dnn_output_val.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    # val_name = './trainData/validseq{}%.pkl'.format(int(args.val_split * 100))

    f = h5py.File(args.path_to_hdf5, "r")
    x = f['tracings']

    n_samples = x.shape[0]
    n_train = math.ceil(n_samples * (1 - args.val_split))

    valid = x[n_train:]
    f.close()

    # Import data
    # val_name = './trainData/validseq{}%'.format(int(args.val_split * 100))
    # f = h5py.File(val_name, "r")
    # seq = f['validation']
    # f.close()
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(valid,  verbose=1)

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")