import tensorflow as tf
import numpy as np
from model import ResNet50
from datalab import DataLabTrain, DataLabTest
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    Y_hat, model_params = ResNet50()
    #Y_hat = tf.sigmoid(Z)

    X = model_params['input']
    Y_true = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    Z = model_params['out']['Z']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_true))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            train_gen = DataLabTrain('./datasets/train_set/').generator()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ix = 0
            for X_true, Y_true_ in train_gen:
                ix += 1
                if ix % 10 == 0:
                    l, _ = sess.run([loss, train_step], feed_dict={X:X_true, Y_true:Y_true_})
                    #acc = np.mean(y.astype('int32') == Y_true_.astype('int32'))
                    print('epoch: ' + str(ix) + ' loss: ' + str(l))
                else:
                    sess.run([train_step], feed_dict={X: X_true, Y_true: Y_true_})

                if ix % 500 == 0:
                    path = './models/model' + (str(ix))
                    os.makedirs(path)
                    saver.save(sess, path + '/model.ckpt')

                if ix == 5000:
                    break
        finally:
            sess.close()


if __name__ == '__main__':
    train()