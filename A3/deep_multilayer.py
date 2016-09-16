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
#def accuracy(predictions,labels):
#	return 100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0]

def generate_parameters(m,n):
	return tf.Variable(tf.random_normal((m,n))),tf.Variable(tf.zeros([n]))

print("Creating tensorflow graph...")
batch_size=128
#train_subset=10000
num_classes=10
num_iterations=10000

hidden_layer_neurons=[784,1024,256]
beta1=.005
beta2=.005
beta3=.005
beta4=.005
sess=tf.Session()
with sess.as_default():
	tf_X_train=tf.placeholder(tf.float32,shape=(None,image_size*image_size))
	tf_X_test=tf.constant(X_test,tf.float32)
	tf_X_val=tf.constant(X_val,tf.float32)
	tf_y_train=tf.placeholder(tf.float32,shape=(None,num_classes))
	
	W01,b01=generate_parameters(image_size*image_size,hidden_layer_neurons[0])
	W12,b12=generate_parameters(hidden_layer_neurons[0],hidden_layer_neurons[1])
	W23,b23=generate_parameters(hidden_layer_neurons[1],hidden_layer_neurons[2])
	W34,b34=generate_parameters(hidden_layer_neurons[2],num_classes)
	keep_prob=tf.placeholder(tf.float32)
	logits1=tf.nn.dropout(tf.nn.tanh(tf.matmul(tf_X_train,W01)+b01),keep_prob)
	logits2=tf.nn.dropout(tf.nn.tanh(tf.matmul(logits1,W12)+b12),keep_prob)
	logits3=tf.nn.dropout(tf.nn.tanh(tf.matmul(logits2,W23)+b23),keep_prob)
	logits_output=tf.matmul(logits3,W34)+b34
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_output,tf_y_train))+beta1*tf.nn.l2_loss(W01)+beta2*tf.nn.l2_loss(W12)+beta3*tf.nn.l2_loss(W23)+beta4*tf.nn.l2_loss(W34)
	global_step=tf.Variable(0)
	learning_rate=tf.train.exponential_decay(.5,global_step,1000,.96,staircase=True)
	trainer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	train_prediction=tf.nn.softmax(logits_output)
	
	correct_prediction=tf.equal(tf.argmax(train_prediction,1),tf.argmax(tf_y_train,1))
	accuracy=100.0*tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print("Running the graph now...")
	sess.run(tf.initialize_all_variables())
	
	
	for iteration in range(num_iterations):
		offset=(batch_size*iteration)%(y_train.shape[0]-batch_size)
		feed_dict_train={tf_X_train:X_train[offset:(offset+batch_size),:],tf_y_train:y_train[offset:(offset+batch_size),:],keep_prob:0.5}
		feed_dict_eval={tf_X_train:X_train[offset:(offset+batch_size),:],tf_y_train:y_train[offset:(offset+batch_size),:],keep_prob:1.0}
		_,l,predictions=sess.run([trainer,loss,train_prediction],feed_dict=feed_dict_train)
		if (iteration%200==0):
			print("----------------------")
			print("Learning rate=%.2f"%learning_rate.eval())
			print("Minibath Loss at step %d: %f" %(iteration,l))
			print("Minibatch Training accuracy: %.1f" %accuracy.eval(feed_dict=feed_dict_eval))
			print("validation accuracy: %.1f" %accuracy.eval(feed_dict={tf_X_train:X_val,tf_y_train:y_val,keep_prob:1.0}))
	print("Test accuracy: %.1f" %accuracy.eval(feed_dict={tf_X_train:X_test,tf_y_train:y_test,keep_prob:1.0}))

