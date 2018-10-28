import tensorflow as tf

sess = tf.Session()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('LFW_model_cloud/model.ckpt-2000.meta')
saver.restore(sess, tf.train.latest_checkpoint('LFW_model_cloud/'))

graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name('x:0')
op_to_restore = graph.get_tensor_by_name("sigmoid_tensor:0")

file = tf.read_file("test_Adam_Freier_0001.jpg")
decoded = tf.image.decode_jpeg(file, channels=3)

result = sess.run(op_to_restore, feed_dict={x: decoded})
