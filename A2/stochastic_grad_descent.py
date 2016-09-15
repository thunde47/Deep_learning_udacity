from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from sklearn.preprocessing import OneHotEncoder
pickle_file='../notMNIST.pickle'

print("Reading the pickled file...")
datasets=[]
with (open("../notMNIST.pickle","rb")) as openfile:
	while True:
		try:
			datasets.append(pickle.load(openfile))
		except EOFError:
			break

print("Flattening 3D data to 2D...")
image_size=28
X_train=datasets[0]['train_dataset'].reshape(-1,image_size*image_size)
X_test=datasets[0]['test_dataset'].reshape(-1,image_size*image_size)
X_val=datasets[0]['valid_dataset'].reshape(-1,image_size*image_size)

print("One hot encoding the class labels...")
#One hot encoding the data
enc=OneHotEncoder()
y_train=enc.fit_transform(datasets[0]['train_labels'].reshape(-1,1)).toarray()
y_test=enc.fit_transform(datasets[0]['test_labels'].reshape(-1,1)).toarray()
y_val=enc.fit_transform(datasets[0]['valid_labels'].reshape(-1,1)).toarray()

#Performing tensorflow logistic regression using simple gradient descent
def accuracy(predictions,labels):
	return 100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0]

print("Creating tensorflow graph...")
train_subset=10000
num_classes=10
num_iterations=3000
batch_size=128
sess=tf.Session()
with sess.as_default():
	tf_X_train=tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
	tf_X_test=tf.constant(X_test,tf.float32)
	tf_X_val=tf.constant(X_val,tf.float32)
	tf_y_train=tf.placeholder(tf.float32,shape=(batch_size,num_classes))
	
	
	W=tf.Variable(tf.random_normal((image_size*image_size,num_classes)))
	b=tf.Variable(tf.zeros([num_classes]))

	logits=tf.matmul(tf_X_train,W)+b
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_y_train))
	trainer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	train_prediction=tf.nn.softmax(logits)
	val_prediction=tf.nn.softmax(tf.matmul(tf_X_val,W)+b)
	test_prediction=tf.nn.softmax(tf.matmul(tf_X_test,W)+b)
	print("Running the graph now...")
	sess.run(tf.initialize_all_variables())
	for iteration in range(num_iterations):
		offset=(batch_size*iteration)%(y_train.shape[0]-batch_size)
		feed_dict={tf_X_train:X_train[offset:(offset+batch_size),:],tf_y_train:y_train[offset:(offset+batch_size),:]}
		_,l,predictions=sess.run([trainer,loss,train_prediction],feed_dict=feed_dict)
		if (iteration%250==0):
			print("Minibath Loss at step %d: %f" %(iteration,l))
			print("Minibatch Training accuracy: %.1f" %accuracy(predictions,y_train[offset:(offset+batch_size),:]))
			print("validation accuracy: %.1f" %accuracy(val_prediction.eval(),y_val))
	print("Test accuracy: %.1f" %accuracy(test_prediction.eval(),y_test))

