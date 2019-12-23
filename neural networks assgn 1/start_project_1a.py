#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
import argparse
import multiprocessing as mp



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


def ffn(batch_size):
    NUM_FEATURES = 21
    NUM_CLASSES = 3
    NUM_HIDDEN = 10

    lr = 0.01
    no_epochs = 5000
    #batch_size = 32
    num_neurons = 10
    seed = 10
    np.random.seed(seed)

    #read train data

    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
    trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int) #trainX=all rows(except first=name) and 0th-20th columns(features),train_Y (d=target output)= all rows (except first=name) of last col
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #normalize data

    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) #create a 2126 x 3 matrix of values=0. 2126=total no. of rows(input data), 3=no. of classes
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix. train_Y.shape[0]=2126

    # initialization routines for bias and weights
    def init_bias(n = 1):
        return(tf.Variable(np.zeros(n), dtype=tf.float32))
        
    def init_weights(n_in=1, n_out=1, logistic=True):
        W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                                high=np.sqrt(6. / (n_in + n_out)),
                                                size=(n_in, n_out)))
        if logistic == True:
            W_values *= 4
        return(tf.Variable(W_values, dtype=tf.float32))


    # experiment with small datasets
    trainX = trainX[:1000] #first 1000 col of data
    trainY = trainY[:1000] #first 1000 col of data

    n = trainX.shape[0] #n=1000

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    #5-fold cross validation
    kf = KFold(n_splits=5) # Define the split - into 5 folds 
    print(kf.get_n_splits(X_train)) # returns the number of splitting iterations in the cross-validator

    print(kf)#KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X_train):
        x_tr, x_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        #print('X TRAIN VALUES:',x_tr)
        #print('X TEST VALUES:',x_te)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)


    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    print(x.shape)
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) #d?


    #Define variables:
    V = init_weights(NUM_HIDDEN, NUM_CLASSES) #weight of output layer
    c = init_bias(NUM_CLASSES) #bias of output layer
    W = init_weights(NUM_FEATURES, NUM_HIDDEN) #weight of hidden layer
    b = init_bias(NUM_HIDDEN) #bias of hidden layer


    z = tf.matmul(x, W) + b #synaptic input to hidden layer
    h = tf.nn.relu(z) #output of hidden layer
    u = tf.matmul(h, V) + c #output of output layer
    y = 

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))

    #optimizer
    grad_u = -(y_ - y)
    grad_V = tf.matmul(tf.transpose(h), grad_u)
    grad_c = tf.reduce_sum(grad_u, axis=0)

    y_h = h*(1-h)
    grad_z = tf.matmul(grad_u, tf.transpose(V))*y_h
    grad_W = tf.matmul(tf.transpose(x), grad_z)
    grad_b = tf.reduce_sum(grad_z, axis=0)

    W_new = W.assign(W - lr*grad_W)
    b_new = b.assign(b - lr*grad_b)
    V_new = V.assign(V - lr*grad_V)
    c_new = c.assign(c - lr*grad_c)

    '''
    # Build the graph for the deep net
            
    weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits  = tf.matmul(x, weights) + biases
    '''

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
    loss = tf.reduce_mean(cross_entropy)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(x_tr)
    idx = np.arange(N)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        
        W_, b_ = sess.run([W, b])
        print('W: {}, b: {}'.format(W_, b_))
        V_, c_ = sess.run([V, c])
        print('V:{}, c:{}'.format(V_, c_))
        
        err=[]
        for i in range(no_epochs):
            train_op.run(feed_dict={x: x_tr, y_: y_tr})
            train_acc.append(accuracy.eval(feed_dict={x: x_tr, y_: y_tr}))
            
            if i == 0:
                z_, h_, y2, cost_, grad_u_, dh_, grad_z_, grad_V_, grad_c_, grad_W_, grad_b_ = sess.run(
                    [ z, h, y, loss, grad_u, y_h, grad_z, grad_V, grad_c, grad_W, grad_b], {x:x_tr, y_:y_tr})
                print('iter: {}'.format(i+1))
                print('z: {}'.format(z_))
                print('h: {}'.format(h_))
                print('y: {}'.format(y2))
                print('grad_u: {}'.format(grad_u_))
                print('dh: {}'.format(dh_))
                print('grad_z:{}'.format(grad_z_))
                print('grad_V:{}'.format(grad_V_))
                print('grad_c:{}'.format(grad_c_))
                print('grad_W:{}'.format(grad_W_))
                print('grad_b:{}'.format(grad_b_))
                print('cost: {}'.format(cost_))
                

            sess.run([W_new, b_new, V_new, c_new], {x:x_tr, y_:y_tr})
            cost_ = sess.run(cost, {x:x_tr, y_:y_tr})
            err.append(cost_)
            
            if i % 100 == 0:
                print('iter %d: accuracy %g'%(i, train_acc[i]))
            

        time_to_update = 0
        for i in range(no_epochs):
            np.random.shuffle(idx)
            x_tr = x_tr[idx]
            y_tr = y_tr[idx]

            t = time.time()
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            time_to_update += time.time() - t
          

            if i%10 == 0:
                test_acc = accuracy.eval(feed_dict={x: x_te, y_: y_te})
                print('batch %d: iter %d, test accuracy %g'%(batch_size, i, test_acc))

        paras = np.zeros(2)
        paras[0] = (time_to_update*1e3)/(no_epochs*(N//batch_size))
        paras[1] = accuracy.eval(feed_dict={x: x_te, y_: y_te})
    return paras


'''
def optimal_batch(batch_size):
    datapoint_size = 1000
    #batch_size = 1000
    steps = 5000
    actual_W1 = 2
    actual_W2 = 5
    actual_b = 7 
    learn_rate = 0.01
    log_file = "/tmp/feature_2_batch_1000"

    # Model linear regression y = Wx + b
    x = tf.placeholder(tf.float32, [None, 2], name="x")
    W = tf.Variable(tf.zeros([2,1]), name="W")
    b = tf.Variable(tf.zeros([1]), name="b")
    with tf.name_scope("Wx_b") as scope:
      product = tf.matmul(x,W)
      y = product + b

    # Add summary ops to collect data
    W_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("y", y)

    y_ = tf.placeholder(tf.float32, [None, 1])

    # Cost function sum((y_-y)**2)
    with tf.name_scope("cost") as scope:
      cost = tf.reduce_mean(tf.square(y_-y))
      cost_sum = tf.summary.scalar("cost", cost)

    # Training using Gradient Descent to minimize cost
    with tf.name_scope("train") as scope:
      train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

    all_xs = []
    all_ys = []
    for i in range(datapoint_size):
      # Create fake data for y = 2.x_1 + 5.x_2 + 7
      x_1 = i%10
      x_2 = np.random.randint(datapoint_size/2)%10
      y = actual_W1 * x_1 + actual_W2 * x_2 + actual_b
      # Create fake data for y = W.x + b where W = [2, 5], b = 7
      all_xs.append([x_1, x_2])
      all_ys.append(y)

    all_xs = np.array(all_xs)
    all_ys = np.transpose([all_ys])

    sess = tf.Session()

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_file, sess.graph_def)

    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(steps):
      if datapoint_size == batch_size:
        batch_start_idx = 0
      elif datapoint_size < batch_size:
        raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
      else:
        batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
      batch_end_idx = batch_start_idx + batch_size
      batch_xs = all_xs[batch_start_idx:batch_end_idx]
      batch_ys = all_ys[batch_start_idx:batch_end_idx]
      xs = np.array(batch_xs)
      ys = np.array(batch_ys)
      all_feed = { x: all_xs, y_: all_ys }
      # Record summary data, and the accuracy every 10 steps
      if i % 10 == 0:
        result = sess.run(merged, feed_dict=all_feed)
        writer.add_summary(result, i)
      else:
        feed = { x: xs, y_: ys }
        sess.run(train_step, feed_dict=feed)
      print("After %d iteration:" % i)
      print("W: %s" % sess.run(W))
      print("b: %f" % sess.run(b))
      print("cost: %f" % sess.run(cost, feed_dict=all_feed))

    
    '''

def main():
  batch_sizes = [4, 8, 16, 32, 64]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  for i in range(5):
      paras = ffn(batch_sizes[i])
      
  paras = np.array(paras)
  
  accuracy, time_update = paras[:,1], paras[:,0]

#  accuracy, time_update = [], []
#  for batch in batch_sizes:
#    test_acc, time_to_update = train(batch)
#    accuracy.append(test_acc)
#    time_update.append(time_to_update)

    # plot learning curves
  plt.figure(1)
  plt.plot(range(len(batch_sizes)), accuracy)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('accuracy')
  plt.title('accuracy vs. batch size')
  plt.savefig('./figures/5.5b_1.png')

  plt.figure(2)
  plt.plot(range(len(batch_sizes)), time_update)
  plt.xticks(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('batch size')
  plt.ylabel('time to update (ms)')
  plt.title('time to update vs. batch size')
  plt.savefig('./figures/5.5b_2.png')
 
#  plt.show()

main()


# plot learning curves
#mln()
#plt.figure(1)
#plt.plot(range(epochs), train_acc)
#plt.xlabel(str(epochs) + ' iterations')
#plt.ylabel('Train accuracy')
#plt.show()


