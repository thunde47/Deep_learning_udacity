from __future__ import print_function
from six.moves import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os.path
import numpy as np

TRAIN_FLAG=False
target_number=2
batch_size = 16
patch_size = 3
depth = 32
num_hidden = 16
num_channels=3
width=128
height=64

achaar_file='svhn.pickle'
with open(achaar_file,'rb') as f:
	save=pickle.load(f)
	all_labels=save['all_labels']
	if TRAIN_FLAG:
		print("Un-achaarifying the training data...")
		train_dataset=save['train_dataset']
		train_target=save['train_target']
	else:
		print("Un-achaarifying the test data...")
		test_dataset=save['test_dataset']
		test_target=save['test_target']
	del save

print("Generating graph...")
graph=tf.Graph()
with graph.as_default():
	#Setting placeholders for datasets
	#tf_num_labels=tf.placeholder(tf.int32)
	for target_number in range(len(all_labels)):
		
		num_labels=all_labels[target_number]
		tf_train_dataset=tf.placeholder(tf.float32, shape=(None, height, width, num_channels))
		tf_train_labels=tf.placeholder(tf.float32, shape=(None, num_labels))

		#Defining layers, weights and biases
		layer1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size, num_channels, depth], stddev=0.1))
		layer1_biases=tf.Variable(tf.zeros([depth]))

		layer2_weights=tf.Variable(tf.truncated_normal([patch_size, patch_size,depth,depth],stddev=0.1))
		layer2_biases=tf.Variable(tf.constant(1.0, shape=[depth]))

		layer3_weights=tf.Variable(tf.truncated_normal([width//4*height//4*depth,num_hidden],stddev=0.1))
		layer3_biases=tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	
		layer4_weights=tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
		layer4_biases=tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
		keep_prob=tf.placeholder(tf.float32)
		saver_global = tf.train.Saver()
		def model(data):
			conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
			hidden1 = tf.nn.relu(conv1 + layer1_biases)
			hidden_pool1=tf.nn.max_pool(hidden1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
			#print("conv layer 1 - ",hidden_pool1.get_shape().as_list())
			conv2 = tf.nn.conv2d(hidden_pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
			hidden2 = tf.nn.relu(conv2 + layer2_biases)
			hidden_pool2=tf.nn.max_pool(hidden2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
			#shape = hidden.get_shape().as_list()
			shape=tf.shape(hidden_pool2)
			#print("conv layer 2 - ",hidden_pool2.get_shape().as_list())
			reshape = tf.reshape(hidden_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
			hidden = tf.nn.dropout(tf.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases),keep_prob)
			#print("fully connected layer",hidden.get_shape().as_list())
			return tf.matmul(hidden, layer4_weights) + layer4_biases

		logits=model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
		global_step=tf.Variable(0)
		learning_rate=tf.train.exponential_decay(.05,global_step,100,.95,staircase=True)
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
		#optimizer=tf.train.AdagradOptimizer(learning_rate=.05,initial_accumulator_value=0.1,use_locking=False)
		optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
		train_prediction=tf.nn.softmax(logits)
		correct_prediction=tf.equal(tf.argmax(train_prediction,1),tf.argmax(tf_train_labels,1))
		accuracy=100.0*tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

		iterations=20
		with tf.Session(graph=graph) as session:
			tf.initialize_all_variables().run()
	
			#if not os.path.exists("parameters.ckpt"):
			if TRAIN_FLAG:
				print("Training network...")
				train_labels=train_target[target_number]
			
				for step in range(iterations):
					offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
					batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
					batch_labels = train_labels[offset:(offset + batch_size), :]
		
					feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
					feed_dict_train_eval={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}
					#feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels, keep_prob:1.0}
					_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_train)
					if (step % 10 == 0):
						print('Minibatch loss at step %d: %f' %(step, l))
						print('Minibatch accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_train_eval))			
			
				saver_local=tf.train.Saver()		
				save_path = saver_local.save(session, "parameters_"+str(target_number)+".ckpt")
				print("Model saved in file: %s" % save_path)
				#print('Test accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_test))
	
			else:
				print("Using trained parameters for prediction...")
				saver_global.restore(session, "parameters_"+str(target_number)+".ckpt")		
				test_labels=test_target[target_number]
				print(test_dataset.shape)
				print(test_labels.shape)
				feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels, keep_prob:1.0}
				test_prediction=session.run([train_prediction],feed_dict=feed_dict_test)
				#print('Test prediction:', test_prediction[0])
				if (target_number>0):
					
					original=np.concatenate((np.array(range(all_labels[target_number]-1)),np.array([10])))
					
				else:
					original=np.array(range(1,all_labels[target_number]+1))
				
				print(original.dot(test_labels[0]))
				print('Test prediction:', test_prediction[0][0], test_labels[0])	
				print('Test accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict_test))
