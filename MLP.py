
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LoadUSPS as usps

class MLP:

    def __init__(self,nodes = 450,lrate=0.05):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.s = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        self.W_h1 = tf.Variable(tf.random_normal([784, nodes]))
        self.b_1 = tf.Variable(tf.random_normal([nodes]))
        self.h1 = tf.nn.sigmoid(tf.matmul(self.x, self.W_h1) + self.b_1)

        self.W_out = tf.Variable(tf.random_normal([nodes, 10]))
        self.b_out = tf.Variable(tf.random_normal([10]))


        #y_ = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)
        self.y_ = tf.matmul(self.h1, self.W_out) + self.b_out
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y)
        #cross_entropy = tf.reduce_sum(- y * tf.log(y_), 1)
        self.loss = tf.reduce_mean(self.cross_entropy)
        #train_step = tf.train.GradientDescentOptimizer(lrate).minimize(loss)
        self.train_step = tf.train.AdamOptimizer(lrate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.x_test = self.mnist.test.images
        self.y_test = self.mnist.test.labels
        self.loadUSPS = usps.LoadUSPS('./proj3_images.zip')
        self.x_usps, self.y_usps = self.loadUSPS.load(one_hot=True)

    # train
    def train(self):
        self.s.run(tf.global_variables_initializer())

        for i in range(15000):
            batch_x, batch_y = self.mnist.train.next_batch(100)
            self.s.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})
            
            if i % 1000 == 0:
                #train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                train_accuracy = self.s.run(self.accuracy,feed_dict={self.x: self.x_test, self.y: self.y_test})
                #print(batch_x.shape, batch_y.shape)
                print('step {0}, training accuracy {1}'.format(i, train_accuracy))    

        print('accuracy of USPS recognition: ', self.s.run(self.accuracy,feed_dict={self.x: self.x_usps, self.y: self.y_usps}))