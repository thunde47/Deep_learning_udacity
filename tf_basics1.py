import tensorflow as tf
w1=tf.ones((2,2))
w2=tf.Variable(tf.random_normal((2,2)),name="ram_weights")
w3=tf.constant(35.0)
with tf.Session() as sess:
	print(sess.run(w1))
	print(sess.run(w3))
	#variables need to be initialized before I can print'hem
	#constants however can be printed direclty
	sess.run(tf.initialize_all_variables())
	print(sess.run(w2))

