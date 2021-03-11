import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
with tf.Session() as sess:
    saver.restore(sess, "./Model/model.ckpt")  # 注意路径写法
    y = []
    sess.run(y,feed_dict={})