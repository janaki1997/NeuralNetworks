#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle



NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 10
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')
    
    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_




def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1 and poo1 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    #Conv 2 and pool 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    #fully connected layer
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    reshape = tf.reshape(pool_2, [-1, dim])
    
    w = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(dim), name="weights3"))
    b = tf.Variable(tf.zeros([300]), name = "biases_3")
    fc1 = tf.nn.relu(tf.matmul(reshape, w) + b, name= "fc1")
	
    #Softmax
    W2 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(dim)), name='weights_4')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_3')
    logits = tf.add(tf.matmul(fc1, W2), b2, name= "softmax_linear")


    return logits


def main():

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    
    logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc=[]
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            training_cost = 0.0

            _, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})

            #for batch_start_idx, batch_end_idx in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            #    training_cost += train( train_x[batch_start_idx:batch_end_idx], train_y[batch_start_idx:batch_end_idx] )

            #training error
            #training_costs = np.append(training_costs, training_cost/(N // batch_size))
            #print('Epoch {}'.format(str(x+1)))

            #test error

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('iter %d: test accuracy %g'%(e, test_acc[e]))

            #entropy
            print('epoch', e, 'entropy', loss_)


    ind = np.random.randint(low=0, high=10000)
    X = trainX[ind,:]
    
    plt.figure()
    plt.gray()
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig('./p1b_2.png')

    

    plt.figure()
    plt.plot(np.arange(epochs), test_acc, label='gradient descent')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.title('Test Error vs Iterations')
    plt.savefig('./1a_test_error_vs_iterations.png')


if __name__ == '__main__':
  main()
