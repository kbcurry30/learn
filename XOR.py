import tensorflow as tf
import numpy as np
 
 #define the data
data_x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype="float")
data_y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype="float")
 
#create model
_x = tf.placeholder("float", [None, 2])
_y = tf.placeholder("float", [None, 1])
_lr = tf.placeholder("float")
W1 = tf.Variable(tf.random_normal([2, 3], 0.1, 0.1))
W2 = tf.Variable(tf.random_normal([3, 1], 0.1, 0.1))
b1 = tf.Variable(tf.random_normal([3], 0.1, 0.1))
b2 = tf.Variable(tf.random_normal([1], 0.1, 0.1))
z1 = tf.matmul(_x, W1) + b1
a1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(a1, W2) + b2
y = tf.nn.sigmoid(z2)
 
#get loss and init
loss_l2 = tf.reduce_sum((_y-y)**2)
train_step = tf.train.GradientDescentOptimizer(_lr).minimize(loss_l2)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
 
#train
errors = []
for epoch in range(10000):
    if epoch <= 2000:
        lr = 1
    elif epoch <=5000:
        lr = 0.1
    else:
        lr = 0.01
    for i in range(4):
        batch_xs, batch_ys = data_x[i:i+1,], data_y[i:i+1,]
        _, loss, zz1, aa1, zz2, yy = sess.run([train_step, loss_l2, z1, a1, z2, y], feed_dict={_x: batch_xs, _y: batch_ys, _lr: lr})
    errors.append(loss)
    print("epoch: %d -->loss: %f--->y: %f"%(epoch, loss, yy))
 
#predict test
def predict(x):
    x = np.array(x, dtype="float32")
    assert x.shape == (1, 2)
    a1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    y = tf.nn.sigmoid(tf.matmul(a1, W2) + b2)
    return y
 
test = predict([[0.1,0.7]])
print(sess.run(test))
