from six.moves import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

achaar_file='svhn.pickle'
with open(achaar_file,'rb') as f:
	save=pickle.load(f)
	train_dataset=save['train_dataset']
	train_target=save['train_target']
	del save

print(train_dataset.shape)

enc=OneHotEncoder()

train_lengths=enc.fit_transform(train_target[0].reshape(-1,1)).toarray()


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 32
num_labels=4
num_channels=3
width=128
height=64

graph=tf.Graph()
with graph.as_default():
	#Setting placeholders for datasets
	tf_train_dataset=tf.placeholder(tf.float32, shape=(batch_size, height, width, num_channels))
	tf_train_labels=tf.placeholder(tf.float32, shape=(batch_size, num_labels))

	#Defining layers, weights and biases
	layer1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size, num_channels, depth], stddev=0.1))
	layer1_biases=tf.Variable(tf.zeros([depth]))

	layer2_weights=tf.Variable(tf.truncated_normal([patch_size, patch_size,depth,depth],stddev=0.1))
	layer2_biases=tf.Variable(tf.zeros([depth]))

	layer3_weights=tf.Variable(tf.truncated_normal([width//4*height//4*depth,num_hidden],stddev=0.1))
	layer3_biases=tf.Variable(tf.zeros([num_hidden]))
	
	layer4_weights=tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
	layer4_biases=tf.Variable(tf.zeros([num_labels]))
	
	def model(data):
		conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer1_biases)
		print("conv layer 1 - ",hidden.get_shape().as_list())
		conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer2_biases)
		shape = hidden.get_shape().as_list()
		print("conv layer 2 - ",shape)
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		print("fully connected layer",hidden.get_shape().as_list())
		return tf.matmul(hidden, layer4_weights) + layer4_biases

	logits=model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	
	train_prediction=tf.nn.softmax(logits)
	correct_prediction=tf.equal(tf.argmax(train_prediction,1),tf.argmax(tf_train_labels,1))
	accuracy=100.0*tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

iterations=1000
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("variables initialized.")
	for step in range(iterations):
		offset = (step * batch_size) % (train_lengths.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_lengths[offset:(offset + batch_size), :]
		
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 10 == 0):
			print('Minibatch loss at step %d: %f' %(step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy.eval(feed_dict=feed_dict))
      #print('Validation accuracy: %.1f%%' % accuracy(
       # valid_prediction.eval(), valid_labels))
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
 
	
	


