# IMPORTS
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Read Data-sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

# Declare a placeholder (inputs any no of MNIST images)
# Each image flattened (2D tensor of Floating point numbers with shape as [None(any length), 784])
x = tf.placeholder(tf.float32, [None, 784])

# Declare a tensor Var(modifiable)
# Weight (Multiply 784-dimensional image vectors to produce 10 dimensional vectors of evidence)
# Bias (10 sized)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implement the model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Placeholder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# Implement the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Train the model (Minimize Cross Entropy, with GradientDescent, with 0.5 alpha)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialise all variables
init = tf.initialize_all_variables()

# Run a Session
sess = tf.Session()
sess.run(init)

# Training begins (1000 times iterated) (Stochastic Gradient Descent Training step by step)
# Feed batch data to replace placeholders
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation of our model
# Where prediction matches truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Accuracy on Test Data
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

