import numpy as np
import os
import pandas as pd

def make_sub(file):
    Y = np.load('out.npy')
    Y = np.clip(Y,0.0125, 0.975)
    li = os.listdir('../Dog-vs-Cat-Tensorflow/datasets/test/')
    li = sorted([int(x.split('.')[0]) for x in li])
    print(len(li))

    res = np.zeros((Y.shape[0], 2))
    for i in range(len(li)):
        res[i, 0] = float(li[i])
        res[i, 1] = Y[i]


    di = {
        'id': res[:, 0].astype('int32'),
        'label': res[:, 1]
    }

    df = pd.DataFrame(di)
    df = df.sort_values(by='id')
    df.to_csv('./submissions/' + file, columns=['id', 'label'], index=False)
