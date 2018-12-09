import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.preprocessing import LabelBinarizer


os.chdir('../data')

filename = 'fs_master_dataset.csv.gz'
df = pd.read_csv(filepath_or_buffer=filename, compression='gzip')

df = utils.shuffle(df, random_state=None, n_samples=None)

hm_epochs = 3
n_classes = 2
batch_size = 128

chunk_size = 251
n_chunks = 2
rnn_size = 512

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('int32', [batch_size])

X = df.drop('label', axis=1, inplace=False)
X = X.astype(float)

Y = df[['label']].astype(int)

# Y = []
#
# for _ in df.index.values:
#     row = df.loc[_]
#     label = [0] * n_classes
#
#     label[int(row['label'])] = 1
#     Y += label
#
# Y = np.array(Y).astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train).tolist()
# y_test = lb.fit_transform(y_test).tolist()


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def next_batch(x, y, batch_num, batch_size):
    assert len(x) == len(y)

    start = batch_num * batch_size
    end = start + batch_size

    if end < len(x):
        return np.array(x[start:end]), np.array(y[start:end])

    return np.array(x[start:]), np.array(y[start:])


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:

    # print(prediction.get_shape())
    # print(y.get_shape())

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(y_train) / batch_size)):
                epoch_x, epoch_y = next_batch(x_train, y_train, _, batch_size)
                epoch_x = epoch_x.reshape(batch_size, n_chunks, chunk_size)
                epoch_y = epoch_y.reshape(batch_size, )

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: np.array(x_test).reshape(-1, n_chunks, chunk_size),
                                          y: np.array(y_test).reshape(-1, batch_size)}))


train_neural_network(x)








