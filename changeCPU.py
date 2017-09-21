import tensorflow as tf


with tf.device('/cpu:0'):
    matrix1 = tf.constant([2 ,2 ,2 ,2 ], shape=[2, 2],name='matrix1',dtype=tf.float32)
    matrix2 = tf.constant([3 ,3 ,3 ,3 ], shape=[2, 2],name='matrix2',dtype=tf.float32)

with tf.device('/gpu:0'):
    product = tf.matmul(matrix1,matrix2)
    
    
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(matrix1))
print(sess.run(matrix2)) 
print(sess.run(product)) 



