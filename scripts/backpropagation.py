import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Multi-class classification
# 10 classes, 0-9

"""
MS Dataset: set of 60,000 training examples of
handwritten digits (0 to 9). All samples are
28 * 28 = 784 pixels.
60,000 training examples + 10,000 testing examples

input layer -> weight -> hidden layer 1 (activation function)
-> weights -> hidden layer 2 (activation function)
-> weights -> output layer

Passing of data straight through of the neural network: feed-forward

compare output to intended output -> cost or loss function (cross entropy)

optimization function (optimizer) -> minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

One_Hot:
0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0]
"""


# Hidden Layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# matrix = height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # Create a tensor using random numbers as weights for the first layer
    # (input_data * weights) + biases
    # Dynamic Neural Network
    # Biases avoid cases that no neuron would fire when inputs are zeros
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases
    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    # Rectified linear function = activation function
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_2, output_layer['weights']) + output_layer['biases']

    # Setting up computation graph modeling
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    # Calculate the difference between the prediction and the known label
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW VERSION:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Synonymous with stochastic gradient descent and AdaGrad
    # default learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Epochs = cycles feed forward + backprop (fixing weights)
    num_epochs = 20
    with tf.Session() as sess:
        # OLD VERSION:
        # sess.run(tf.initialize_all_variables())
        # NEW VERSION:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0

            # Calculate how many cycles are needed to complete entire batch
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # Optimize cost with x and y
                # x -> data from neural_network_model [Line 48]
                # y -> labels from cost [Line 88]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                # How much cost is reduced during each epoch
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', num_epochs, 'with loss:', epoch_loss)

        # Determine whether maximum of each one_hot are identical
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)




