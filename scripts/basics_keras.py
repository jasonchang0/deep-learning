import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization of data inputs to between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Feedforward neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# 128 units/neurons per layer with rectified linear activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# output layer with number of classifications
# use softmax for probability distribution function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Epoch = 'full pass' through the entire training dataset
model.fit(x_train, y_train, epochs=3)

# Evaluate the output of sample data with constructed model
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)
print(val_loss)
print(val_acc)

# To save the trained model
model.save('digit_reader.model')
input_model = tf.keras.models.load_model('digit_reader.model')

predictions = input_model.predict([x_test])

print(predictions)
print(np.argmax(predictions[0]))


# graph in binary color map
print(x_test[0])
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()





