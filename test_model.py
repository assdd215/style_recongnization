import tensorflow as tf


def simple_model():
    w1 = tf.placeholder(dtype=tf.float32,name="w1")
    w2 = tf.placeholder(dtype=tf.float32,name="w2")
    b1 = tf.Variable(2.0,name="bias")
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.save(sess,"./checkpoint_dir/MyModel",global_step=1000)

def load_model():
    sess = tf.Session()
    saver = tf.train.import_meta_graph("./checkpoint_dir/MyModel-1000.meta")
    saver.restore(sess,tf.train.latest_checkpoint("./checkpoint_dir"))

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    feed_dict = {w1:13,w2:17}
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    print(sess.run(op_to_restore,feed_dict))
