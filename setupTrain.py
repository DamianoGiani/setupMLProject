import h5py
import numpy as np


f = h5py.File('exams_part15.hdf5', 'r')
traces_ids = np.array(f['exam_id'])
trainTracings = f['tracings']
traces_ids = np.delete(traces_ids, -1, 0)
trainBatch=trainTracings[:20000, :, :]
traces_ids2, trainBatch2 = zip(*sorted(zip(traces_ids, trainBatch)))
f = h5py.File("train15.hdf5", "w")
f.create_dataset("tracings", data=trainBatch2)
f.close()