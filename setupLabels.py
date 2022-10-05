import numpy as np
import pandas as pd

labels = pd.read_csv('examslabels.csv').values
labels=np.delete(labels,[1,2,3,10,11,12,13],axis=1)
labelsFrame = pd.DataFrame(labels)
labelsSplit=[]
for i in range(18):
  labelsSplit.append(labelsFrame.loc[labelsFrame[7] == 'exams_part' + str(i) + '.hdf5'])
for i in range(18):
  labelsSplit[i].sort_values([0], axis=0, ascending=[True], inplace=True)
  labelsSplit[i] = labelsSplit[i].drop([7],axis=1)
  labelsSplit[i][:] = labelsSplit[i][:].astype(int)
  labelsSplit[i].drop(0, inplace=True, axis=1)
  labelsSplit[i].to_csv('labels/labels'+str(i)+'.csv',index=False,header=False,sep=',')