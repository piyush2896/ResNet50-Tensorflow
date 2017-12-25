import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

Y = np.load('out.npy')
Y = np.clip(Y, 0.0125, 0.975)
path = '../Dog-vs-Cat-Tensorflow/datasets/test/'
li = os.listdir(path)
li = sorted([int(x.split('.')[0]) for x in li])
li = [str(x) + '.jpg' for x in li]

for i in range(50):
    y = plt.figure(i)
    plt.imshow(cv2.cvtColor(cv2.imread(path+li[i]), cv2.COLOR_BGR2RGB))
    plt.title(Y[i])
    #y.axes.get_xasix().set_visible(False)
    #y.axes.get_yasix().set_visible(False)
    plt.show()