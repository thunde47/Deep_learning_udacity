from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from sklearn.preprocessing import OneHotEncoder
pickle_file='../notMNIST.pickle'

'''
print("Reading the pickled file...")
datasets=[]
with (open("../notMNIST.pickle","rb")) as openfile:
	while True:
		try:
			datasets.append(pickle.load(openfile))
		except EOFError:
			break
'''
with open(pickle_file,'rb') as f:
	save=pickle.load(f)
	train_dataset=save['train_dataset']
	train_labels=save['train_labels']
	valid_dataset=save['valid_dataset']
	valid_labels=save['valid_labels']
	test_dataset=save['test_dataset']
	test_labels=save['test_labels']
	del save
	

image_size=28
num_channels=1
train_dataset=train_dataset.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
test_dataset=test_dataset.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
valid_dataset=valid_dataset.reshape(-1,image_size,image_size,num_channels).astype(np.float32)
test_dataset=test_dataset[0:5000,:]
valid_dataset=valid_dataset[0:5000,:]
print("One hot encoding the class labels...")
#One hot encoding the data
enc=OneHotEncoder()
train_labels=enc.fit_transform(train_labels.reshape(-1,1)).toarray()
test_labels=enc.fit_transform(test_labels.reshape(-1,1)).toarray()
valid_labels=enc.fit_transform(valid_labels.reshape(-1,1)).toarray()
test_labels=test_labels[0:5000,:]
valid_labels=valid_labels[0:5000,:]

'''
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape,y_test.shape)
'''
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_labels=10



graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(None, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32,shape=(None, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  keep_prob=tf.placeholder(tf.float32)
  def model(data):
    
    conv1 = tf.nn.conv2d(data, layer1_weights,strides=[1,1,1,1],padding='SAME')
    hidden = tf.nn.relu(conv1 + layer1_biases)
    hidden_pool1=tf.nn.max_pool(hidden,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #print("conv layer 1 - ",hidden_pool1.get_shape().as_list())
    conv2 = tf.nn.conv2d(hidden_pool1, layer2_weights,strides=[1,1,1,1],padding='SAME')
    hidden = tf.nn.relu(conv2 + layer2_biases)
    hidden_pool2=tf.nn.max_pool(hidden,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #print("conv layer 1 - ",hidden_pool2.get_shape().as_list())
    shape = tf.shape(hidden_pool2)    
    reshape = tf.reshape(hidden_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases),keep_prob)
    #print("fully connected layer - ",hidden.get_shape().as_list())
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  correct_prediction=tf.equal(tf.argmax(train_prediction,1),tf.argmax(tf_train_labels,1))
  accuracy=100.0*tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 

num_steps=1001
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized all the variables')
  
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:0.5}
    feed_dict_train_eval = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:1.0}
    feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels,keep_prob:1.0}
    feed_dict_valid = {tf_train_dataset : valid_dataset, tf_train_labels : valid_labels,keep_prob:1.0}

    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_train)
    if (step % 50 == 0):
      print("-----------------------------------")
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_train_eval))
      print('Validation accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_valid))
  print('Test accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_test))
	




