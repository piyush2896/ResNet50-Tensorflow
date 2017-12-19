import tensorflow as tf
import numpy as np
from datalab import DataLabTest
from model import ResNet50
import matplotlib.pyplot as plt
from make_file import make_sub


def predict(model_path, batch_size):
    Y_hat, model_params = ResNet50()
    X = model_params['input']

    saver = tf.train.Saver()

    test_gen = DataLabTest('./datasets/test_set/').generator()

    Y = []
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for i in range(12500//batch_size+1):
            y = sess.run(Y_hat, feed_dict={X: next(test_gen)})
            print(y.shape, end='   ')
            Y.append(y[:, 1])
            print(len(Y))
    Y = np.concatenate(Y)

    print(Y.shape)
    return Y


if __name__ == '__main__':
    Y = predict('./models/model5000/model.ckpt', 32)
    np.save('out.npy', Y)
    make_sub('sub_1.csv')
