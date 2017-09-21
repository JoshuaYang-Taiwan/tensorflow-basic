import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

##Initializer is important
x.initializer.run()

sub = tf.subtract(x,a)
print(sub.eval()) ## =>[-2,-1]

sess.close()

